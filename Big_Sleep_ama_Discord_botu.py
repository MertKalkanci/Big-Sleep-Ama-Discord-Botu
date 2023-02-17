import nest_asyncio 
nest_asyncio.apply()

import discord
from discord.ext.commands import Bot
import random

from torch import randint
from tqdm.notebook import trange
from IPython.display import Image, display
import random
from big_sleep import Imagine
import warnings

async def generate_img(TEXT,EPOCH,IT, CTX):
  global STOPPER
  old_msg = await CTX.send(f'Generating: "{TEXT}", Epochs: "{EPOCH}",Iterations: "{IT}"')

  _EPOCH = int(EPOCH)
  _IT = int(IT)
  SAVE_EVERY = _IT / 2
  SAVE_PROGRESS = True 
  LEARNING_RATE = 5e-2 
  ITERATIONS = 1050
  SEED = 0

  model = Imagine(
    text = TEXT,
    save_every = SAVE_EVERY,
    lr = LEARNING_RATE,
    iterations = ITERATIONS,
    save_progress = SAVE_PROGRESS,
    seed = SEED
  )
  
  with open("./PP.png", "rb") as fh:
    f = discord.File(fh, filename="./PP.png")
    
  msg = await CTX.send(content="Initialised Model",file=f)

  filename = model.text.replace(' ', '_')
  print("\n")
  print(f'./{filename}.png')
  fullname = f'./{filename}.png'

  for epoch in trange(_EPOCH, desc = 'epochs'):
    
    print(f"Stopper : {STOPPER}")
    if STOPPER:
      STOPPER = False
      break

    for i in trange(_IT, desc = 'iteration'):
        model.train_step(epoch, i)

    with open(fullname, "rb") as fh:
      f = discord.File(fh, filename=fullname)
    my_list = [f]
    await msg.edit(content=f"Epoch: {epoch + 1} in {_EPOCH}",attachments=my_list)




  with open(fullname, "rb") as fh:
          f = discord.File(fh, filename=fullname)
  my_list = [f]
  await msg.edit(content=TEXT,attachments=my_list)
  await old_msg.delete()

  return

global STOPPER
STOPPER = False

warnings.filterwarnings('ignore')

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = Bot(intents=intents,command_prefix="$")

@bot.command()
async def generate(ctx, epoch, iter ,*args):
  text = ""
  for i in args:
    text += i + " "
  text = text.rstrip(text[-1])


  await generate_img(text,epoch, iter, ctx)
   

@bot.command()
async def sj(ctx, *args):
  text = ""
  for i in args:
    text += i
    
  await ctx.send(text)  

@bot.command()
async def stop(ctx):
  global STOPPER
  STOPPER=True
  print("Stopping")
  

@bot.event
async def on_connect():
  print("Bot Connected")
    
bot.run('YOUR TOKEN HERE', log_handler=None)