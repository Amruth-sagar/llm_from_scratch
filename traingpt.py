import torch
from llm_from_scratch.gpt.model.custom_gpt import CustomGPTwithRoPE
from llm_from_scratch.tokenizer.bpe_tokenizer import BytelevelBPE
from llm_from_scratch.data.gpt_dataset import GPTDataset, get_dataloader
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from torch.optim import AdamW
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from llm_from_scratch.utils.loss_and_metric import cross_entropy

def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def make_deterministic_minimal(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(checkpoint_name, ckpt_dir, model, optimizer, optimizer_steps, scaler, global_step, prev_val_loss, args, GPT_CFG):
    save_path = f'{ckpt_dir}/{checkpoint_name}.pth'

    # Since we are saving a DDP wrapped model, the actual
    # model is in model.module
    model_state_dict = model.module.state_dict()
    optim_state_dict = optimizer.state_dict()
    scaler_state_dict = scaler.state_dict()

    torch.save({
        "model_state_dict":model_state_dict,
        "optim_state_dict":optim_state_dict,
        "scaler_state_dict":scaler_state_dict,
        "global_step":global_step,
        "optimizer_steps":optimizer_steps,
        "prev_val_loss":prev_val_loss,
        "args":vars(args),
        "GPT_CFG":GPT_CFG,
    }, save_path)



def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--max_train_steps', type=int, required=True)
    argument_parser.add_argument('--val_after_k_optim_steps', type=int,  required=True)
    argument_parser.add_argument('--lr', type=float,  required=True)
    argument_parser.add_argument('--accum_grad_after_k_steps', type=int,  required=True)
    argument_parser.add_argument('--val_batches_per_device', type=int,  required=True)
    argument_parser.add_argument('--batch_size_per_device', type=int,  required=True)
    argument_parser.add_argument('--metric_log_dir', type=str,  required=True)
    argument_parser.add_argument('--save_ckpt_after_k_steps', type=int,  required=True)
    argument_parser.add_argument('--start_saving_ckpt_after_k_steps', type=int,  required=True)
    argument_parser.add_argument('--ckpt_dir', type=str,  required=True)
    argument_parser.add_argument('--tokenizer_state', type=str,  required=True)
    argument_parser.add_argument('--clean_data_dir', type=str,  required=True)
    argument_parser.add_argument('--random_seed', type=int,  required=True)
    argument_parser.add_argument('--num_docs_for_validation', type=int,  required=True)
    argument_parser.add_argument('--num_dl_workers', type=int,  required=True)
    argument_parser.add_argument('--resume_from_ckpt', type=str, default=None)


    args = argument_parser.parse_args()

    make_deterministic_minimal(args.random_seed)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.metric_log_dir, exist_ok=True)

    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=f"{args.metric_log_dir}")

    tokenizer = BytelevelBPE()
    tokenizer.load(args.tokenizer_state)
    eos_token_id = tokenizer.encode('<eos>')[0]     

    GPT_CFG = {
        "vocab_size": len(tokenizer.vocab),
        "embed_dim": 768,
        "context_len": 2048,
        "max_seq_len": 2048,
        "drop_p_after_embed": 0.05,
        "drop_p_for_mmha": 0.05,
        "drop_p_post_mmha": 0.05,
        "num_tf_blocks": 16,
        "num_heads": 12,    # head_dim = 64
        "qkv_bias": False,
        "rope_base":10000
    }

    model = CustomGPTwithRoPE(cfg=GPT_CFG)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    scaler = torch.amp.GradScaler()


    # random_seed + rank, to avoid same batches created 
    # by dataloader in all processes.
    train_dataset = GPTDataset(
        clean_data_dir=args.clean_data_dir,
        ctx_len=GPT_CFG["context_len"],
        eos_token_id=eos_token_id,
        split='TRAIN',
        num_docs_for_validation=args.num_docs_for_validation,
        random_seed=args.random_seed + rank,
        return_doc_ids=True
    )

    val_dataset = GPTDataset(
        clean_data_dir=args.clean_data_dir,
        ctx_len=GPT_CFG["context_len"],
        eos_token_id=eos_token_id,
        split='VAL',
        num_docs_for_validation=args.num_docs_for_validation,
        random_seed=args.random_seed + rank,
        return_doc_ids=True
    )

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size_per_device,
        num_workers=args.num_dl_workers,
        random_seed=args.random_seed + rank
    )

    train_dataloader_iter = iter(train_dataloader)
    
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size_per_device,
        num_workers=args.num_dl_workers,
        random_seed=args.random_seed + rank
    )
    val_dataloader_iter = iter(val_dataloader)

    global_step = 0
    optimizer_steps = 0
    prev_val_loss = torch.inf
    optimizer.zero_grad(set_to_none=True)

    if args.resume_from_ckpt is not None:
        map_location = {'cuda:0':f'cuda:{local_rank}'}
        ckpt = torch.load(args.resume_from_ckpt, map_location)

        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])

        global_step = ckpt['global_step']
        optimizer_steps = ckpt['optimizer_steps']
        prev_val_loss = ckpt['prev_val_loss']

    # If we are resuming from a ckpt
    max_train_steps = global_step + args.max_train_steps
    

    while global_step < max_train_steps:
        model.train()

        input_tokens, output_tokens, doc_ids = next(train_dataloader_iter)
        input_tokens, output_tokens, doc_ids = input_tokens.to(device), output_tokens.to(device), doc_ids.to(device)
        
        if (global_step + 1) % args.accum_grad_after_k_steps != 0:
            with model.no_sync():
                # autocast to make forward pass faster, by controlling precision
                # based on the type of operation
                with torch.autocast(device_type='cuda'):
                    model_output = model(input_tokens, doc_ids=doc_ids)
                    loss = cross_entropy(model_output, output_tokens)
                    loss = loss / args.accum_grad_after_k_steps
                
                scaler.scale(loss).backward()

        else:
            with torch.autocast(device_type='cuda'):
                model_output = model(input_tokens, doc_ids=doc_ids)
                loss = cross_entropy(model_output, output_tokens)
                loss = loss / args.accum_grad_after_k_steps
            
            # since there is no no_sync context, the gradients are
            # synchronized
            scaler.scale(loss).backward()

            # to log unscaled train loss.
            train_loss = loss.detach() * args.accum_grad_after_k_steps

            if rank == 0:
                writer.add_scalar("loss/train", train_loss, global_step+1)
            
            # Gradient clipping after unscaled gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            optimizer_steps += 1

        if optimizer_steps > 0 and optimizer_steps % args.val_after_k_optim_steps == 0:
            
            model.eval()
            val_loss = torch.zeros(1, device=device)

            with torch.no_grad():
                for _ in range(args.val_batches_per_device):
                    input_tokens, output_tokens, doc_ids = next(val_dataloader_iter)
                    input_tokens, output_tokens, doc_ids = input_tokens.to(device), output_tokens.to(device), doc_ids.to(device)
                    with torch.autocast(device_type='cuda'):
                        model_output = model(input_tokens, doc_ids=doc_ids)
                        loss = cross_entropy(model_output, output_tokens)
                        val_loss += loss.detach()
            
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= (args.val_batches_per_device * world_size)

            if rank == 0:
                writer.add_scalar("loss/val", val_loss.item(), global_step+1)
            
            if prev_val_loss > val_loss:
                prev_val_loss = val_loss
                if rank == 0:
                    checkpoint_name = "GPTwithRoPE_best_val"
                    save_checkpoint(checkpoint_name, args.ckpt_dir, model, optimizer, optimizer_steps, scaler, global_step, prev_val_loss, args, GPT_CFG)
            
            # reset to 0, else it will keep running validation loops till next increment happens
            optimizer_steps = 0
            

        if (global_step + 1) % args.save_ckpt_after_k_steps == 0 and (global_step+1 ) >= args.start_saving_ckpt_after_k_steps:
            dist.barrier()
            if rank == 0:
                checkpoint_name = f"GPTwithRoPE_step_{global_step+1}_time_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                save_checkpoint(checkpoint_name, args.ckpt_dir, model, optimizer, optimizer_steps, scaler, global_step, prev_val_loss, args, GPT_CFG)
            dist.barrier()
        
        global_step += 1

    cleanup_distributed()
    if rank == 0:
        writer.close()


if __name__ == "__main__":
    main()
