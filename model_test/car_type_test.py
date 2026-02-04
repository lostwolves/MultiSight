import json
import os
import time
import requests
import base64

from typing import Union, Dict
from mimetypes import guess_type
from tqdm import tqdm

os.environ['OPENAI_API_KEY']="YOUR_API_KEY_HERE"

class GPTS:
    def __init__(self,
                 api_key:str=None,
                 model_name:str="gpt-4-turbo-preview",
                 temperature:float=0.5,
                 top_p:float=1.0,
                 max_tokens:int=2000):
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        assert self.api_key, "API Key is missing!"

        self.supported_model_names = ["gpt-3.5-turbo", 
                                      "gpt-4-turbo-preview", 
                                      "gpt-4-turbo", 
                                      "gpt-4-vision-preview", 
                                      "gpt-4-all", 
                                      "GPTS", 
                                      "gpt-4-1106-preview",
                                      "gpt-4-0125-preview",
                                      "gpt-4-turbo-2024-04-09",
                                      "gpt-4o",]

        assert model_name in self.supported_model_names, f"Model name should be one of {self.supported_model_names}"

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.default_prompt = "You are a useful assistant who can efficiently complete user-specified tasks or answer user questions well."
        self.default_sleep_time = 2
        self.max_retries = 5

        # self.url="https://reverse.onechat.fun/v1/chat/completions"
        self.url="https://chatapi.onechats.top/v1/chat/completions"

    def set_attr(self,name:str,value)->None:
        if hasattr(self,name):
            setattr(self,name,value)
        else:
            raise AttributeError(f"{name} does not exist in the class")
    
    def get_headers(self)->Dict[str,str]:
        return {
            "Content-Type": "application/json",
            "Authorization":f"Bearer {self.api_key}"
        }
    
    def get_payload(self,prompt:str=None)->Dict:
        if not prompt:
            prompt = self.default_prompt
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
        }

    def chat(self,question:str, image_files:Union[str,list]=None, prompt:str=None, history:dict=None):
        payload=history or self.get_payload(prompt)
        
        payload["messages"].append(
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":question
                    }
                ]
            }
        )

        if image_files:
            if "gpt-4" not in self.model_name:
                raise ValueError("Image input is only supported for GPT-4 models")
            if isinstance(image_files,str):
                image_files = [image_files]
            for image_file in image_files:
                mime_type,_ = guess_type(image_file)
                strImage=self.encode_image(image_file)
                payload["messages"][-1]["content"].append(
                    {
                        "type":"image_url",
                        "image_url":{
                            "url":f"data:{mime_type};base64,{strImage}"
                        }
                    }
                )

        retries = 0
        while retries<self.max_retries:
            response = requests.post(self.url,json=payload,headers=self.get_headers())
            if response.status_code == 200:
                break
            else:
                retries+=1
                time.sleep(self.default_sleep_time)

        if retries==self.max_retries:
            raise Exception("Max retries reached.")
        
        output = self.parse_response(response)
        payload["messages"].append(
            {
                "role":"assistant",
                "content":[
                    {
                        "type":"text",
                        "text":output
                    }
                ]
            }
        )
        history = payload

        return output, history

    def parse_response(self,response):
        return response.json()["choices"][0]["message"]["content"]

        
    def encode_image(self,image_file):
        with open(image_file,"rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

def gpt_test_model_answer(gpt, image_list, save_name, true_label):
    image_num = len(image_list)
    task_prompt = "I will conduct a fine-grained car recognition task based on multi-view images. You will need to determine the specific car model in the images, such as Volkswagen, Chevrolet, Toyota, etc., based on the provided multi-view images. I will provide you with one image from a different angle at a time. Please do not provide an answer until I tell you, 'This is the last angle of the image. Please give the recognition result without any explanation.' In other cases, I will only say, 'This is an image from one angle of the target vehicle,' and you should simply reply, 'Received.' Are you ready?"
    image_prompt1 = "This is an image from one angle of the target vehicle."
    image_prompt2 = "This is the last angle of the image. Please give the recognition result without any explanation."
    start_time = time.time()
    output, history = gpt.chat(task_prompt)
    print(output)
    ## Input multi-view images sequentially to obtain the inferred car model
    for i in range(image_num):
        if i == image_num - 1:
            output, history = gpt.chat(image_prompt2, image_files=image_list[i], history = history)
            # print(output)
            answer_dict = {"result": output, "label": true_label}
            with open(save_name, "w", encoding="utf-8") as fp:
                json.dump(answer_dict, fp, indent=4, ensure_ascii=False)
        else:
            output, history = gpt.chat(image_prompt1, image_files=image_list[i], history = history)
            # print(output)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time spent running the code: {elapsed_time:.4f} seconds")
    answer_dict = {"result": output, "label": true_label}
    with open(save_name, "w", encoding="utf-8") as fp:
        json.dump(answer_dict, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    
    cfg=dict(
        model_name="gpt-4o",
        # model_name="gpt-4-turbo", 
        temperature=0,
        top_p=0.9,
        max_tokens=300,
    )
    gpt = GPTS(**cfg)
    
    ## Get path dict
    cargroup_read_dir = 'cartype_image_grouped'
    cargroup_label_dir = 'cartype_label'
    answer_save_dir = 'cartype_image_grouped_answer'
    if not os.path.exists(answer_save_dir):
        os.makedirs(answer_save_dir)
    for test_group in tqdm(os.listdir(cargroup_read_dir)):
        mark = test_group.split('_')[0]
        image_file = [os.path.join(cargroup_read_dir, test_group, name) for name in os.listdir(os.path.join(cargroup_read_dir, test_group)) if name.endswith('.jpg')]
        json_score_save_dir = os.path.join(answer_save_dir, test_group + '_ans.json')
        if os.path.exists(json_score_save_dir):
            continue
        label_dir = os.path.join(cargroup_label_dir, mark + '.json')
        with open(label_dir, 'r', encoding='utf-8') as file:
            data = json.load(file)
            true_label = data[mark]
        gpt_test_model_answer(gpt, image_file, json_score_save_dir, true_label)