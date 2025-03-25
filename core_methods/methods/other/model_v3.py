import requests
 
def chat_with_gpt(prompt,img_url):
    api_key = "YOUR_API_KEY"
    model = "gpt-4o"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":img_url
                        }
                    }
                ]
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            } 
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
 
    else:
        return f"Error: {response.status_code}, {response.text}"
 
if __name__=="__main":
    img_url="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
    prompt="图片里面有什么"
    response = chat_with_gpt(prompt,img_url) 
    print(response)
 
    #response=这张图片展示了一位坐在沙滩上的女性和她的狗狗。她们正在互动，狗狗举起一只前爪，她伸出手与狗狗互动。背景是一个波光粼粼的大海，天空是明亮的日落时分，整个场景显得温馨而美好。