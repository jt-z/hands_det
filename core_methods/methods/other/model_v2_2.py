import openai
import base64
from openai import OpenAI


import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Initialize OpenAI client
def initialize_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)

# Helper function to encode image as base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Create the request message
def create_shadow_puppet_message(img_url):
    return [
        {
            "role": "system",
            "content": "You are an expert assistant that specializes in analyzing shadow puppets. Your task is to identify the animals that shadow puppets resemble and provide similarity scores between 0 and 100. Always respond concisely, returning only the requested information in the specified format."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze the shadow puppet. Identify the two animals that most closely resemble the shadow puppet from the following list: cow, rabbit, wolf, chicken, bird. Provide a similarity score between 0 and 100 for each. The output should be formatted as follows: 'animal_1': <animal name>, 'score_1': <similarity score>, 'animal_2': <animal name>, 'score_2': <similarity score>. Return only these four fields in this format, without any additional explanation."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    }
                }
            ]
        }
    ]

# Send request to OpenAI API
def send_shadow_puppet_request(client, img_url, model="gpt-4-turbo"):
    messages = create_shadow_puppet_message(img_url)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

# Process response from OpenAI API
def process_response(response):
    reply = ""
    for res in response:
        content = res.choices[0].delta.content
        if content:
            reply += content
    return reply

# Main function to analyze the shadow puppet
def analyze_shadow_puppet(client, img_url=None, model="gpt-4-turbo"): 

    # Send the request and process the response
    response = send_shadow_puppet_request(client, img_url, model)
    result = process_response(response)

    return result

def show_image_result(image_data, result):
    img = Image.open(BytesIO(image_data))
    plt.imshow(img)
    plt.axis('off')  # Hide axes for clarity

    # Add text on the image
    plt.text(10, 10, f"结果: {result}", fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

    plt.show()


# Example usage
api_key = 'sk-TwVvabBw8ntN11o13a1d948f9aD8440f8807FdE0F963E657'
base_url = 'https://api.bianxie.ai/v1'
client = initialize_client(api_key, base_url)

# image_path = "../assets/bwolf.png"  # Local image path
# img_url = 'https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/bwolf.png'  # URL to the image

 

# # Analyze the shadow puppet with image path or image URL
# result = analyze_shadow_puppet(client, img_url=img_url)  # You can also use img_url
# print('Result:', result)



# Function to download and display image from a URL
def download_and_display_image(img_url):
    response = requests.get(img_url)
    if response.status_code == 200:
        img_data = response.content
        # img = Image.open(BytesIO(img_data))
        # # Display the image using matplotlib
        # plt.imshow(img)
        # plt.axis('off')  # Hide axes for clarity
        # plt.show()
        return img_data  # Return the image data
    else:
        raise Exception(f"Failed to download image from {img_url}")

# Example usage
img_urls = [
    'https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/20241013224424.png',
    'https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/bchicken.png',
    'https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/bwolf.png',
    'https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/20241013222029.png',
    'https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/20241013222145.png',
    'https://jt-pub-images.oss-cn-beijing.aliyuncs.com/picgo/ee.png' 
]

for img_url in img_urls:
    try:
        # Download and display each image
        image_data = download_and_display_image(img_url)
        
        # Analyze the shadow puppet using the image URL
        result = analyze_shadow_puppet(client, img_url=img_url)  # Assuming analyze_shadow_puppet is predefined
        print('Result:', result)
        show_image_result(image_data,result)
    
    except Exception as e:
        print(f"Error: {e}")
