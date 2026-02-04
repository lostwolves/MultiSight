import json
import os
import time
import requests
import base64

from typing import Union, Dict
from mimetypes import guess_type
from tqdm import tqdm

os.environ['OPENAI_API_KEY']="XXX" # Replace with your actual API key

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

def gpt_test_model_answer(gpt, drone_image_file, satellite_image_file, json_score_save_dir, true_label):
    
    X = len(drone_image_file)
    Y = len(satellite_image_file)
    task_prompt = "I will conduct a task to locate targets based on multi-view images from a drone. You will need to determine which of the satellite image options the building in the multi-view drone images belongs to. First, I will provide you with one drone image from a different viewpoint at a time, and I will inform you with 'There are a total of M drone images, and this is the number N. ' Each time, you only need to respond with 'Drone image received!' After all M drone images have been input, I will then provide you with a satellite image option one at a time, and I will tell you 'There are a total of P satellite images, and this is the number Q.' Each time, you only need to respond with 'Received option Q!' Finally, when I input the last satellite image option, I will inform you with 'There are a total of P satellite images, and this is the number P. Could you please identify which satellite image option the building in the M drone images belongs to?' Without giving any explanation, please respond with 'Option X.' Do you understand?"
    image_prompt1 = f"There are a total of {X} drone images, and this is the number X1."
    image_prompt2 = f"There are a total of {Y} satellite images, and this is the number Y1."
    image_prompt3 = f"There are a total of {Y} satellite images, and this is the number {Y}. Could you please identify which satellite image option the building in the {X} drone images belongs to?"
    output, history = gpt.chat(task_prompt)
    print(output)
    ## First, input the multi-view drone images of the buildings.
    for i in range(X):
        mid_image_prompt1 = image_prompt1.replace("X1", str(i + 1))
        output, history = gpt.chat(mid_image_prompt1, image_files=drone_image_file[i], history = history)
        print(output)
    ## Then, input the option for satellite imagery.
    for i in range(Y):
        if i == Y - 1:
            output, history = gpt.chat(image_prompt3, image_files=satellite_image_file[str(Y)], history = history)
            print(output)
            answer_dict = {"result": output, "label": true_label}
            with open(json_score_save_dir, "w", encoding="utf-8") as fp:
                json.dump(answer_dict, fp, indent=4, ensure_ascii=False)
        else:
            mid_image_prompt2 = image_prompt2.replace("Y1", str(i + 1))
            output, history = gpt.chat(mid_image_prompt2, image_files=satellite_image_file[str(i + 1)], history = history)
            print(output)

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
    drone_build_group_dir = 'gallery_drone_group'
    satellite_gallery_dir = 'gallery_satellite_group'
    answer_save_dir = 'test_answer/GPT-4o'
    
    if not os.path.exists(answer_save_dir):
        os.makedirs(answer_save_dir)
    for test_group in tqdm(os.listdir(drone_build_group_dir)):
        mark = test_group.split('_')[0]
        drone_image_file = [os.path.join(drone_build_group_dir, test_group, name) for name in os.listdir(os.path.join(drone_build_group_dir, test_group)) if name.endswith('.jpeg')]
        satellite_image_file = {}
        for i, name in enumerate(os.listdir(os.path.join(satellite_gallery_dir, mark))):
            satellite_image_file[str(i+1)] = os.path.join(satellite_gallery_dir, mark, name)
            name_choice = name.replace('.jpg', '')
            if name_choice == mark:
                true_label = i + 1
        json_score_save_dir = os.path.join(answer_save_dir, test_group + '_ans.json')
        if os.path.exists(json_score_save_dir):
            continue
        gpt_test_model_answer(gpt, drone_image_file, satellite_image_file, json_score_save_dir, true_label)