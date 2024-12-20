from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

def do_sft(ds_train, ds_eval, max_seq_length, output_dir, epochs, resume_from_checkpoint=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )


    train_args = TrainingArguments(
        auto_find_batch_size=True,
        learning_rate=3e-4,
        lr_scheduler_type="linear",

        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,

        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir=output_dir+'/training',
        seed=42,
        eval_delay=0,
        evaluation_strategy="epoch",
    )
    train_args.set_logging(
        report_to= ["tensorboard"],
        steps = 10,
        strategy = "steps",
    )
    train_args.set_save(
        strategy="epoch"
    )
    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=train_args,
        dataset_text_field="messages",
    )

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(output_dir=output_dir+"/model")