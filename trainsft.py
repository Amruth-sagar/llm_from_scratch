import torch
from llm_from_scratch.gpt.model.custom_gpt import CustomGPTwithRoPE
from llm_from_scratch.tokenizer.bpe_tokenizer import BytelevelBPE
from llm_from_scratch.data.sft_dataset import SFTDataset, get_sft_dataloader
import argparse
import torch
from torch.optim import AdamW
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from llm_from_scratch.utils.loss_and_metric import cross_entropy



def make_deterministic_minimal(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(checkpoint_name, ckpt_dir, model, optimizer, global_step, prev_val_loss, args, GPT_CFG):
    save_path = f'{ckpt_dir}/{checkpoint_name}.pth'

    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()

    torch.save({
        "model_state_dict":model_state_dict,
        "optim_state_dict":optim_state_dict,
        "global_step":global_step,
        "prev_val_loss":prev_val_loss,
        "args":vars(args),
        "GPT_CFG":GPT_CFG,
    }, save_path)



def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--num_epochs', type=int, required=True)
    argument_parser.add_argument('--val_every_k_steps', type=int,  required=True)
    argument_parser.add_argument('--lr', type=float,  required=True)
    argument_parser.add_argument('--batch_size', type=int,  required=True)
    argument_parser.add_argument('--metric_log_dir', type=str,  required=True)
    argument_parser.add_argument('--save_ckpt_after_k_steps', type=int,  required=True)
    argument_parser.add_argument('--ckpt_dir', type=str,  required=True)
    argument_parser.add_argument('--tokenizer_state_sft', type=str,  required=True)
    argument_parser.add_argument('--sft_data_dir', type=str,  required=True)
    argument_parser.add_argument('--random_seed', type=int,  required=True)
    argument_parser.add_argument('--num_samples_for_validation', type=int,  required=True)
    argument_parser.add_argument('--num_dl_workers', type=int,  required=True)
    argument_parser.add_argument('--load_ckpt', type=str, required=True)


    args = argument_parser.parse_args()

    make_deterministic_minimal(args.random_seed)

    device = torch.device("cuda:0")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.metric_log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=f"{args.metric_log_dir}")

    tokenizer = BytelevelBPE()
    tokenizer.load(args.tokenizer_state_sft)
    eos_token_id = tokenizer.encode('<eos>')[0]
    pad_token_id = tokenizer.encode('<pad>')[0]
    user_token_id = tokenizer.encode('<user>')[0]
    asst_token_id = tokenizer.encode('<assistant>')[0]     


    ckpt = torch.load(args.load_ckpt)
    GPT_CFG = ckpt['GPT_CFG']

    # random_seed + rank, to avoid same batches created 
    # by dataloader in all processes.
    train_dataset = SFTDataset(
        clean_data_dir=args.sft_data_dir,
        ctx_len=GPT_CFG["context_len"],
        pad_token_id=pad_token_id,
        user_token_id=user_token_id,
        asst_token_id=asst_token_id,
        eos_token_id=eos_token_id,
        split='TRAIN',
        num_samples_for_validation=args.num_samples_for_validation,
        random_seed=args.random_seed,
    )

    val_dataset = SFTDataset(
        clean_data_dir=args.sft_data_dir,
        ctx_len=GPT_CFG["context_len"],
        pad_token_id=pad_token_id,
        user_token_id=user_token_id,
        asst_token_id=asst_token_id,
        eos_token_id=eos_token_id,
        split='VAL',
        num_samples_for_validation=args.num_samples_for_validation,
        random_seed=args.random_seed,
    )

    train_dataloader = get_sft_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_dl_workers,
        drop_last=True
    )
    
    val_dataloader = get_sft_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_dl_workers,
        drop_last=True
    )

    global_step = 0
    prev_val_loss = torch.inf


    model = CustomGPTwithRoPE(cfg=GPT_CFG)
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad(set_to_none=True)

    model.resize_embedding_and_lm_head(len(tokenizer.vocab))

    # Updating the GPT_CFG's vocab size since the old one 
    # is without the newly added special tokens.
    GPT_CFG['vocab_size'] = len(tokenizer.vocab)

    for epoch in range(args.num_epochs):
        for batch in train_dataloader:
            model.train()

            input_tokens, output_tokens, pad_token_mask = batch
            input_tokens, output_tokens, pad_token_mask = input_tokens.to(device), output_tokens.to(device), pad_token_mask.to(device)

            model_output = model(input_tokens, attn_mask=pad_token_mask)
            loss = cross_entropy(model_output, output_tokens)           

            loss.backward()

            train_loss = loss.detach()

            writer.add_scalar("loss/train", train_loss, global_step+1)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (global_step + 1) % args.val_every_k_steps == 0:
                
                model.eval()
                val_loss = torch.zeros(1, device=device)

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        input_tokens, output_tokens, pad_token_mask = val_batch
                        input_tokens, output_tokens, pad_token_mask = input_tokens.to(device), output_tokens.to(device), pad_token_mask.to(device)

                        model_output = model(input_tokens, attn_mask=pad_token_mask)
                        loss = cross_entropy(model_output, output_tokens)
                        val_loss += loss.detach()
                
                val_loss /= (len(val_dataloader))

                writer.add_scalar("loss/val", val_loss.item(), global_step+1)
                
                if prev_val_loss > val_loss:
                    prev_val_loss = val_loss
                    checkpoint_name = "SFT_GPTwithRoPE_best_val"
                    save_checkpoint(checkpoint_name, args.ckpt_dir, model, optimizer, global_step, prev_val_loss, args, GPT_CFG)


            if (global_step + 1) % args.save_ckpt_after_k_steps == 0:
                checkpoint_name = f"SFT_GPTwithRoPE_step_{global_step+1}_time_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                save_checkpoint(checkpoint_name, args.ckpt_dir, model, optimizer, global_step, prev_val_loss, args, GPT_CFG)
            
            global_step += 1
    
    writer.close()


if __name__ == "__main__":
    main()
