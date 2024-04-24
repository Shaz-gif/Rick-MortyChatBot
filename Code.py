
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_size = "small"

tokenizer = AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{model_size}")
model = AutoModelForCausalLM.from_pretrained(f"microsoft/DialoGPT-{model_size}")

"""## Chat with the untrained model"""

def chat(model, tokenizer, trained=False):
    print("type \"q\" to quit. Automatically quits after 5 messages")

    for step in range(5):
        message = input("MESSAGE: ")

        if message in ["", "q"]:  # if the user doesn't wanna talk
            break

        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids


        # generated a response while limiting the total chat history to 1000 tokens,
        if (trained):
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature = 0.8,
            )
        else:
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

chat(model, tokenizer)

"""It's capable of holding a conversation, but doesn't resemble Rick Sanchez at all yet

## Configuring the model
"""

import glob, logging, os, pickle, random, re, torch, pandas as pd, numpy as np
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm.notebook import tqdm, trange
from pathlib import Path
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

# Args to allow for easy convertion of python script to notebook
class Args():
    def __init__(self):
        self.output_dir = f'output-{model_size}'
        self.model_type = 'gpt2'
        self.model_name_or_path = f'microsoft/DialoGPT-{model_size}'
        self.config_name = f'microsoft/DialoGPT-{model_size}'
        self.tokenizer_name = f'microsoft/DialoGPT-{model_size}'
        self.cache_dir = 'cached'
        self.block_size = 512
        self.per_gpu_train_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 40  # 3
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 1000
        self.save_total_limit = None
        self.seed = 42
        self.local_rank = -1

args = Args()



df = pd.read_csv("../input/harry-potter-final-data/final_data.csv")
"""
contexted = []
n = 7

for i in range(n, len(data['line'])):
  row = []
  prev = i - 1 - n
  for j in range(i, prev, -1):
    row.append(data['line'][j])
  contexted.append(row)

columns = ['response'] + ['context '+str(i+1) for i in range(n)]
df = pd.DataFrame.from_records(contexted, columns=columns)
df.head(5)"""

df.head()

"""We want the model to be aware of previous messages from the dialogue to help it decide what to say next. Here we modify the dataset to include context from 7 previous messages.

### Formatting the data and defining some helper functions
We need to construct the data in the right format so the bot can interpret it properly. To do this we're adding special characters like the 'end of string' charater
"""

def construct_conv(row, tokenizer, eos = True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv

def load_and_cache_examples(args, tokenizer, df_trn):
    return ConversationDataset(tokenizer, args, df_trn)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        directory = args.cache_dir
        cached_features_file = os.path.join(directory, args.model_type + "_cached_lm_" + str(block_size))

        logger.info("Creating features from dataset file at %s", directory)
        self.examples = []
        for _, row in df.iterrows():
            conv = construct_conv(row, tokenizer)
            self.examples.append(conv)

        logger.info("Saving features into cached file %s", cached_features_file)
        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

"""# Training

Now, this is quite a hefty chunk of code but don't worry you don't need to understant it yet, we can cover this in later tutorials
"""

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate, drop_last = True
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info("*** Running trainng, Num examples = %d, Num Epochs = %d ***", len(train_dataset), args.num_train_epochs)

    global_step, epochs_trained = 0, 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            inputs, labels = (batch, batch)
            if inputs.shape[1] > 1024: continue
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

    tb_writer.close()

    return global_step, tr_loss / global_step

"""# Main Runner

Here we're simply setting up the logger and starting the training!
"""

def main(df_trn):
    args = Args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s", args.local_rank, device, args.n_gpu)

    set_seed(args) # Set seed

    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=False, config=config, cache_dir=args.cache_dir)
    model.to(args.device)

    # Training
    train_dataset = load_and_cache_examples(args, tokenizer, df_trn)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = AutoModelForCausalLM.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model.to(args.device)

config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=False, config=config, cache_dir=args.cache_dir)
train_dataset = load_and_cache_examples(args, tokenizer, df)

df

"""# Lets Run it!
This should take around 5 minutes so you might as well go grab a cup of coffee ☕️
"""

main(df)

"""# Chatting with the trained bot"""

tokenizer = AutoTokenizer.from_pretrained(f'microsoft/DialoGPT-{model_size}')
model = AutoModelForCausalLM.from_pretrained(f'output-{model_size}')
chat(model, tokenizer, trained=True)

"""# Conclusion

Any dialogue can be used to trian this bot, it just needs to be in the right format.

We can also try changing the training config. For example, the context length (`n`), or any of the arguments in the `Args` class.
"""
