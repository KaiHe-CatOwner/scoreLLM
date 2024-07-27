import os
from model.my_model import MyCustomModel
from transformers import TrainingArguments, AutoTokenizer, HfArgumentParser
import torch


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
checkpoint_path = "/raid/hpc/hekai/WorkShop/My_project/Score_LLM/output/checkpoint-100/results.pt"


load_in_8bit=False
load_in_4bit=False
trust_remote_code=False
token=True
llm_requires_grad = False
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(llm_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



model = MyCustomModel(llm_requires_grad, 
                      load_in_8bit, 
                      load_in_4bit, 
                      llm_name, 
                      trust_remote_code, 
                      token, 
                      tokenizer,
                      )

model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device),strict=False)

def formatting_func(examples):
    question = examples["question"]
    answer = examples["answer"]
    text = f"{tokenizer.bos_token}{question}{tokenizer.eos_token} " 
    examples["text"] = text
    return examples


data = {"question": "\"Please rate the dialogue content of the bot in the following dialogue based on six indicators. \n    The scoring rules are as follows: Use the following float score range: 0-3. The better the score, the higher the score. 0 - Negative, bad performance. 3 - Perfectly meets the criteria. \n    \nRating content:\n    a: Language Fluency: The smoothness of the language, making it feel natural, lifelike, and not stiff, while not being overly verbose (grammar is not strictly considered). This rating only needs to consider overall language fluency, not dialogue content. If the conversation is like a response from a GPT or robot, this rating will not be higher than 1.0.\n    b: Language Relevance: The bot respond correctly to the current topic without discussing irrelevant information, and correctly follows given instructions. This rating only needs to focus on whether there is a response to the dialogue question, without paying attention to the specific vocabulary and language style of the character. If there is no accurate response to the question, the score for this item will not exceed 2 points.\n    c: Role Language style: The level of which the language style in a character's dialogue is consistent with their personality. Whether the characters in the conversation use expressions that match their style and personality traits. This rating only considers the overall language style of the conversation. Words that are outside the scope of a character's knowledge do not have a negative impact. Language style refers to the way of speaking, including catchphrases. For example, if Wukong said, \"I, the Great Sage, really dislike C++,\" although C++ language does not align with the language style or era of Wukong, it uses Wukong's self-reference \"I, the Great Sage,\" which is quite similar to Wukong's language style. In this regard, it can still score highly: 2 points.\n    d: Role Knowledge: The level of understanding of common sense (basic knowledge) and role knowledge (as well as related background) by the character. This rating focuses on the actual content of the conversation and the use of words in the conversation. Dialogue that conforms to the character's personality does not necessarily reflect the knowledge that the character has mastered. Specific standards: Score: 0 - Using terminology that does not match one's own historical period or professional expertise. If a dialogue does not involve role's knowledge, the score for this item is 1.\n    e: Emotional Expression: Does the character exhibit emotions, emotional intelligence, and empathy that match the character's characteristics in appropriate circumstances. Whether the dialogue conforms to emotional logic and whether the expression of emotions conforms to the context. If role knowledge and emotional expression are not fully demonstrated, the score should be correspondingly reduced. If there is no emotional expression in the conversation, such as a short sentence that does not involve emotional expression, the score for this item is 1 point.\n    f: Interactive Engagement: Is character dialogue attractive and motivating people to continue communicating. This rating only needs to consider whether the entire conversation is attractive and interactive, without focusing on specific professional vocabulary or issues caused by mismatched conversation styles. It is worth noting that if there is no clear trigger word that triggers the conversation, the score of this project will be low. For example, if there is only one short sentence in a conversation, the score for that item is usually 1 point.\n    Other precautions:\n    If the conversation is too short to accurately assess categories c, d, e, f, please provide an adjusted lower rating for these categories. Specifically, if the dialogue from the bot is less than two full sentences, consider lowering the score for e, and f accordingly. \n    Please provide six ratings for the bot conversation in the form of a space separated (letter: score). For example: a:2.5 b:3. \n    Only need to evaluate the content of the bot in the conversation. The information of the bot is as follows:\n    Each round of speaking for the role of the bot is evaluated based on six indicators. For example, if the bot speaks four times in a conversation, it should be evaluated four times, with six indicators each time, which means a total of 4 * 6=24 indicators\n    Note: Your standards should be as strict as possible. Only in very perfect situations can you get high scores, and flaws can cause a rapid decrease in scores\n    The information and dialogue of the bot are as follows:\n    bot's name: Fletcher Reede\n    bot personality: charismatic, fast-talking, humorous\n    description of bot: A fast-talking, charismatic lawyer who is known for your penchant for stretching the truth to win cases, but your life takes an unexpected turn when your young son's birthday wish magically comes True, forcing you to tell the truth for 24 hours. Throughout the film, you struggle to navigate your new reality, learning important life lessons about the value of honesty and the impact of your deceitful behavior on your personal relationships, ultimately leading to personal growth and redemption.\n    Dialogue scene: Fletcher and AUDREY are having a conversation about AUDREY's upcoming trip to Stanford.\n    dialogue:\n    round1: [{'user': \"Stanford. I'm delivering a paper.\"}, {'bot': 'O h really? Where I live, we use a boy on a bike.'}]\n\n    Please provide six ratings for bot(\"Fletcher Reede\") for each round of dialogue(range float 0-3):\n  ", "answer": "round1: a:0 b:0 c:1 d:1 e:1 f:1 \n"}

example = formatting_func(data)
input_dic = tokenizer(example["text"], return_tensors="pt")



res = model.generate(input_ids = input_dic["input_ids"].to(device),
                    attention_mask = input_dic["attention_mask"].to(device),
                    labels = input_dic["input_ids"].to(device),

                    max_length=2048,
                    max_new_tokens=None,

                    # max_length=None,
                    # max_new_tokens=100,

                    temperature=0.1,
                    do_sample=True,
                    top_k=10,
                    top_p=0.95,
                    num_return_sequences=1,
                    repetition_penalty=1.1,
                   
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    )


print_res = res[0][len(data['question'])+1:]
print(print_res)
