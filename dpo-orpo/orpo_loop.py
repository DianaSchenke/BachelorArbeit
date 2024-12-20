from trl import ORPOTrainer, ORPOConfig

def do_orpo(model,  tokenizer, output_dir, ds, epochs):
    train_args = ORPOConfig(
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
    orpo_trainer = ORPOTrainer(
        model = model,
        args = train_args,
        train_dataset = ds,
        tokenizer = tokenizer,
    )
    orpo_trainer.train()
    orpo_trainer.save_model(output_dir=output_dir+"/model")