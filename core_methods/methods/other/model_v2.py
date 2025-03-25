import openai
import base64
from openai import OpenAI


client = OpenAI(
    api_key='sk-TwVvabBw8ntN11o13a1d948f9aD8440f8807FdE0F963E657',  
    base_url='https://api.bianxie.ai/v1'#可根据镜像站修改
)

#图片转base64函数
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
 
#输入图片路径
image_path = "../assets/bwolf.png"


#原图片转base64
base64_image = encode_image(image_path)

img_url="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
img_url='https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/bchicken.png'
img_url='https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/bwolf.png'
img_url='https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/20241013222029.png'
img_url='https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/20241013222145.png'
img_url='https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/ee.png'



#提交信息至GPT4o
response = client.chat.completions.create(
    model="gpt-4-turbo",#选择模型
    messages=[
        {
        "role": "system",
        "content": "You are an expert assistant that specializes in analyzing shadow puppets. Your task is to identify the animals that shadow puppets resemble and provide similarity scores between 0 and 100. Always respond concisely, returning only the requested information in the specified format."
        },
        {
            "role": "user",
            "content":[
            {
          "type": "text",
          "text": "Analyze the shadow puppet. Identify the two animals that most closely resemble the shadow puppet from the following list: cow, rabbit, wolf, chicken, crocodile. Provide a similarity score between 0 and 100 for each. The output should be formatted as follows: 'animal_1': <animal name>, 'score_1': <similarity score>, 'animal_2': <animal name>, 'score_2': <similarity score>. Return only these four fields in this format, without any additional explanation.",
        },
                    {
          "type": "image_url",
          "image_url":{
            # "url": f"data:image/jpeg;base64,{base64_image}"
            "url":img_url
          }
        },
        ]
        }
    ],
    stream=True,
)

reply = ""
for res in response:
    content = res.choices[0].delta.content
    if content:
        reply += content
        print(content)

print('reply:',reply)