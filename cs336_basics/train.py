import argparse
import torch
import os
import numpy as np
from model import TransformerLM
from training_utils import (
    AdamW,
    load_checkpoint,
    save_checkpoint,
    load_data,
    gradient_clipping,
    cosine_lr_schedule,
    cross_entropy,
)
import time
import math
from pathlib import Path

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model.")

    # Data and Checkpointing
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")

    # Model Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum context length for the model.")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of the inner feed-forward layer.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--theta", type=float, default=10000.0, help="Theta value for RoPE positional embeddings.")

    # Training Hyperparameters
    parser.add_argument("--max_learning_rate", type=float, default=1e-3, help="Maximum learning rate.")
    parser.add_argument("--min_learning_rate", type=float, default=1e-4, help="Minimum learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations for LR scheduler.")
    parser.add_argument(
        "--cosine_iters", type=int, default=None, help="Number of iterations for the cosine decay cycle."
    )

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total number of training steps.")
    parser.add_argument("--eval_interval", type=int, default=500, help="Interval for evaluation and logging.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to train on (e.g., "cpu", "cuda", "mps").',
    )

    args = parser.parse_args()

    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_steps

    print("--- Configuration ---")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("---------------------\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = args.device

    # --- Memory-efficient Data Loading ---
    print("Loading data...")
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")
    print(f"Train data loaded with {len(train_data):,} tokens.")
    print(f"Validation data loaded with {len(val_data):,} tokens.\n")

    # --- Model and Optimizer Initialization ---
    print("Initializing model and optimizer...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.\n")

    # --- Checkpoint Loading ---
    start_iter = 0
    checkpoint_path = Path(args.checkpoint_dir) / "latest_checkpoint.pth"
    if checkpoint_path.exists():
        try:
            start_iter = load_checkpoint(checkpoint_path, model, optimizer)
        except Exception as e:
            print(f"Could not load checkpoint. Starting from scratch. Error: {e}")

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # --- Training Loop ---
    print("Starting training loop...")
    start_time = time.time()

    for it in range(start_iter, args.max_steps):
        # --- Learning Rate Scheduling ---
        lr = cosine_lr_schedule(
            it, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Evaluation ---
        if it % args.eval_interval == 0 or it == args.max_steps - 1:
            model.eval()
            val_loss_accum = 0.0
            eval_iters = 100
            with torch.no_grad():
                for _ in range(eval_iters):
                    x, y = load_data(val_data, args.batch_size, args.context_length, device)
                    logits = model(x)
                    loss = cross_entropy(logits, y)
                    val_loss_accum += loss.items()

            avg_val_loss = val_loss_accum / eval_iters
            perplexity = math.exp(avg_val_loss)

            current_time = time.time()
            elapsed_time = current_time - start_time

            print(
                f"Step {it:>{len(str(args.max_steps))}} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f} | LR: {lr:.6f} | Time: {elapsed_time:.2f}s"
            )

            save_checkpoint(model, optimizer, it, checkpoint_path)
            model.train()

        x, y = load_data(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

    print("\nTraining finished.")
    final_checkpoint_path = Path(args.checkpoint_dir) / f"final_model_step_{args.max_steps}.pth"
    save_checkpoint(model, optimizer, args.max_steps, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
