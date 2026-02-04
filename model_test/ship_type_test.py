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

def gpt_test_model_answer(gpt, image_list, save_name, true_label, main_label):
    image_num = len(image_list)
    task_prompt = "I will conduct a fine-grained ship recognition task based on multi-view images. These images are taken from different perspectives for the same ship target, please give the fine-grained category identification results of this ship. You need to note that you are not only required to determine the category of the ship targets, such as aircraft carrier, destroyer, or frigate, but also to provide its specific name. For instance, if you identify it as an American aircraft carrier with the hull number 76, then it is the USS Ronald Reagan CVN76, so you should only return 'USS_Ronald_Reagan_CVN76' without any explanation. I will provide you with one image from a different angle at a time. Please do not provide an answer until I tell you, 'This is the last angle of the image. Please give the recognition result without any explanation.' In other cases, I will only say, 'This is an image from one angle of the target ship,' and you should simply reply, 'Received.' Are you ready?"
    image_prompt1 = "This is an image from one angle of the target ship."
    image_prompt2 = "This is the last angle of the image. Please give the recognition result without any explanation."
    start_time = time.time()
    output, history = gpt.chat(task_prompt)
    print(output)
    ## Input multi-view images sequentially to obtain the inferred ship type
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
    answer_dict = {"result": output, "label": true_label, "main_label": main_label}
    with open(save_name, "w", encoding="utf-8") as fp:
        json.dump(answer_dict, fp, indent=4, ensure_ascii=False)

def get_folder_names(path):
    """
        Get the names of all folders under the specified path.
        :param path: Target directory path
        :return: List of folder names
    """
    try:
        # Get the names of all files and folders in the specified path
        all_items = os.listdir(path)
        
        # Filter out directories (keep only folders)
        folder_names = [item for item in all_items if os.path.isdir(os.path.join(path, item))]
        
        return folder_names
    except FileNotFoundError:
        print(f"The path {path} does not exist")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def find_matching_element(element_list, input_string):
    """
        Check if the input string contains any element from the given list, and return the first matched element.
        :param element_list: List of elements to match against
        :param input_string: Input string to search in
        :return: The matched element if found, otherwise None
    """
    for element in element_list:
        if element in input_string:
            return element
    return None

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
    shipgroup_read_dir = 'multiview_ship\Image_group'
    test_ship_dir = get_folder_names(shipgroup_read_dir)
    answer_save_dir = 'test_answer\GPT-4o'
    if not os.path.exists(answer_save_dir):
        os.makedirs(answer_save_dir)
    for mid_dir in test_ship_dir:
        mid_path = os.path.join(shipgroup_read_dir, mid_dir)
        main_label = find_matching_element(["Aircraft_Carrier", "Destroyer", "Frigate"], mid_dir)
        for test_group in tqdm(os.listdir(mid_path)):
            mark = test_group.split('_group_')[0]
            true_label = mark.replace("_" + mark.split('_')[-1], "")
            image_file = [os.path.join(mid_path, test_group, name) for name in os.listdir(os.path.join(mid_path, test_group)) if name.endswith('.jpg')]
            json_score_save_dir = os.path.join(answer_save_dir, test_group + '_ans.json')
            if os.path.exists(json_score_save_dir):
                continue
            gpt_test_model_answer(gpt, image_file, json_score_save_dir, true_label, main_label)