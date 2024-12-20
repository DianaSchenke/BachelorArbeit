from transformers import TrainingArguments
from trl import DPOTrainer

def do_dpo(model,  tokenizer, output_dir, ds, epochs):
    train_args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = epochs,
        bf16 = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir=output_dir+'/training',
    )
    train_args.set_logging(
        report_to= ["tensorboard"],
        steps = 10,
        strategy = "steps",
    )
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = train_args,
        beta = 0.1,
        train_dataset = ds,
        tokenizer = tokenizer,
        max_length = 2048,
        max_prompt_length = 512,
    )
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir=output_dir+"/model")