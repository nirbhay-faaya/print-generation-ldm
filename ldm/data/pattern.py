import os
import numpy as np
import PIL
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import re
import string

def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img

class PatternTextToImageDataset():
    def __init__(self,
                csv_files_folder: str,
                images_folder: str,
                csv_to_img_mapping: {str,str},
                additional_paths: [str],
                img_transforms: transforms):

        """
        csv_files_format: title, description, keywords, id
        """
        
        self.csv_files_path = csv_files_folder
        self.img_path = images_folder
        self.img_transforms = img_transforms
        self.csv_file_list = os.listdir(self.csv_files_path)


        self.data = []

        for i in self.csv_file_list:
            if i in csv_to_img_mapping:
                path = os.path.join(self.csv_files_path, i)
                data = pd.read_csv(path)
                data['combined_text'] = self._combine_text(data)
                data['combined_text'] = data['combined_text'].apply(lambda x: self._preprocess(x))
                data.iloc[:,3] = data.iloc[:,3].apply(lambda x: os.path.join(self.img_path, csv_to_img_mapping[i], str(x)) + '.jpg')
                self.data.extend(data.iloc[:,[4,3]].values)

        for additional_path in additional_paths:
            self.data.extend(self._process_additional_paths(additional_path))

        self.len_pre_process = len(self.data)

        self.data = list(filter(lambda x:os.path.exists(x[1]), self.data))
        
    def _combine_text(self, data):
        data.iloc[:,0] = data.iloc[:,0].astype(str).apply(lambda x: '' if x == 'nan' else x)
        data.iloc[:,1] = data.iloc[:,1].astype(str).apply(lambda x: '' if x == 'nan' else x)
        data.iloc[:,2] = data.iloc[:,2].astype(str).apply(lambda x: '' if x == 'nan' else x)
        return data.iloc[:,0] + data.iloc[:,1] + data.iloc[:,2] 

    def _process_additional_paths(self, path):

        # getting csv file and image folders path
        csv_files_path = os.path.join(path, 'annotations')
        images_folders_path = os.path.join(path, 'images')
        # getting list of csv files
        csv_files_list = os.listdir(csv_files_path)

        # creating dict of text: image-path 
        data_temp: [tuple(str,str)] = []
        for csv_files in csv_files_list:
            read_file = pd.read_csv(os.path.join(csv_files_path, csv_files)) # reading csv file
            read_file = read_file.drop_duplicates(subset=['Id'])
            read_file = read_file.dropna()
            read_file['combined_text'] = self._combine_text(read_file) # combine first three columns
            read_file['combined_text'] = read_file['combined_text'].apply(lambda x: self._preprocess(x)) # process them using regex
            regex = r'\d+'
            image_folder_id = re.search(regex, csv_files).group(0) # extract folder_id from csv file path
            image_folder = os.path.join(images_folders_path, image_folder_id)
            read_file.iloc[:,3] = read_file.iloc[:, 3].apply(
                lambda x: os.path.join(image_folder, str(int(x))) + '.jpg'
            )

            data_temp.extend(read_file.iloc[:,[4,3]].values)

        return data_temp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, image_path = self.data[idx]

        loaded_img = get_imgs(image_path, transform=self.img_transforms)
        loaded_img = loaded_img.permute(1,2,0)
        # tokens = clip.tokenize(text,truncate=True)[0]
        input_data = {}
        input_data["image"] = loaded_img
        input_data["caption"] = text

        return input_data

    def _preprocess(self, text):
        text = text.lower()#converting string to lowercase
        text = text.replace("illustrations", "")
        text = text.replace("illustration", "")
        res1 = re.sub(r'((http|https)://|www.).+?(\s|$)',' ',text)#removing links
        res2 = re.sub(f'[{string.punctuation}]+',' ',res1)#removing non english and special characters
        res3 = re.sub(r'[^a-z0-9A-Z\s]+',' ',res2)#removing anyother that is not consider in above
        res4 = re.sub(r'(\n)+',' ',res3)#removing all new line characters
        res = re.sub(r'\s{2,}',' ',res4)#remove all the one or more consecutive occurance of sapce
        res = res.strip()
        return res


class PrintGenerationData(PatternTextToImageDataset):
    def __init__(self, **kwargs):
        root_path = kwargs['root_path']
        image_folder = os.path.join(root_path,"dataset/dataset2/refined")
        csv_folder = os.path.join(root_path,"dataset/dataset2/tags_out")
        csv_img_mapping = {
            'istock_African pattern fabric.csv':'African pattern fabric',
            'istock_batik motif fabric.csv':'batik motif fabric',
            'istock_Ajrakh print.csv':'Ajrakh print',
            'istock_camouflage pattern fabric.csv':'camouflage pattern fabric',
            'istock_Animal print fabric.csv':'Animal print fabric',
            'istock_checkered pattern fabric.csv':'checkered pattern fabric',
            'istock_Daisy illustration fabric.csv':'Daisy illustration fabric',
            'istock_herringbone seamless pattern textile.csv':'herringbone seamless pattern textile',
            'istock_Floral print fabric.csv':'Floral print fabric', 
            'istock_kaleidoscope mandala art fabric.csv':'kaleidoscope mandala art fabric',
            'istock_Houndstooth seamless pattern.csv':'Houndstooth seamless pattern', 
            'istock_kaleidoscope pattern fabric.csv':'kaleidoscope pattern fabric',
            'istock_Madras Check fabric.csv':'Madras Check fabric',
            'istock_madhubani art print.csv':'madhubani art print',
            'istock_Polka dots fabric.csv':'Polka dots fabric',   
            'istock_mandala art fabric.csv':'mandala art fabric',
            'istock_Red checkered tablecloths patterns.csv':'Red checkered tablecloths patterns',
            'istock_psychedelic print fabric.csv':'psychedelic print fabric',
            'istock_Rose print fabric.csv':'Rose print fabric',    
            'istock_shibori pattern.csv':'shibori pattern',
            'istock_Tropical Print fabric.csv':'Tropical Print fabric',
            'istock_tribal seamless pattern.csv':'tribal seamless pattern'
        }

        additional_paths = [
            os.path.join(root_path,'dataset/dataset5'),
            os.path.join(root_path,'dataset/dataset4'),
            os.path.join(root_path,'dataset/dataset6')
        ]

        img_transforms = transforms.Compose([
            transforms.Resize((kwargs["size"],kwargs["size"])),
            transforms.ToTensor()
        ])

        super().__init__(
            csv_folder,
            image_folder,
            csv_img_mapping,
            additional_paths,
            img_transforms
        )


class UpdatedPatternTextToImageDataset():
    def __init__(self,
                csv_files_folder: str,
                images_folder: str,
                img_transforms: transforms):

        """
        csv_files_format: image_name, caption, mini_caption, timetaken
        """
        
        self.csv_files_path = csv_files_folder
        self.img_path = images_folder
        self.img_transforms = img_transforms

        self.error_threshold = 80

        self.csv_file_paths_map = {}
        self.folder_dfs(
            self.csv_files_path,
            ['csv'],
            self.csv_file_paths_map
        )

        self.image_file_path_map = {}
        for data_folders in self.img_path:
            self.folder_dfs(
                data_folders,
                ['jpg', 'png', 'jpeg', 'JPG', 'PNG'],
                self.image_file_path_map
            )  

        erroneous_captions = {}

        img_caption_arr = None
        for csv_file_name, csv_path in self.csv_file_paths_map.items():
            csv_data = pd.read_csv(csv_path)
            csv_data['mini-caption'] = csv_data['mini-caption'].apply(lambda x: self._preprocess(str(x)))
            # erroneous_captions[csv_file_name] = csv_data.loc[
            #     csv_data['mini-caption'].str.len() < self.error_threshold, 
            #     ['image_name', 'mini-caption']].values

            csv_data = csv_data.loc[
                csv_data['mini-caption'].str.len() > self.error_threshold, 
                ['image_name', 'mini-caption']].values

            if img_caption_arr is None:
                img_caption_arr = csv_data
            else:
                img_caption_arr = np.concatenate([img_caption_arr, csv_data], axis=0)

        img_caption_arr = dict(img_caption_arr)
        self.image_file_path_map = dict(filter(
            lambda x: x[0] in img_caption_arr,
            self.image_file_path_map.items()
        ))

        # print(len(self.image_file_path_map))
        # print(len(img_caption_arr))

        # print(erroneous_captions['data_2_tail.csv'])

        # img_caption_arr = list(img_caption_arr.items())
        # for i in range(10):
        #     image, caption = img_caption_arr[i]
        #     print(f'{image} => {caption}')

        self.data = list(map(
            lambda x: (img_caption_arr[x[0]], x[1]),
            self.image_file_path_map.items()
        ))

        self.len_pre_process = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, image_path = self.data[idx]

        loaded_img = get_imgs(image_path, transform=self.img_transforms)
        loaded_img = loaded_img.permute(1,2,0)

        input_data = {}
        input_data["image"] = loaded_img
        input_data["caption"] = text

        return input_data

    def folder_dfs(self, 
        dir_path: str, 
        allowed_ext: [str], 
        final_paths_dict: {str, str}):

        cur_dirs = os.listdir(dir_path)

        for dirs in cur_dirs:
            cur_path = os.path.join(dir_path, dirs)
            if os.path.isdir(cur_path):
                self.folder_dfs(
                    cur_path,
                    allowed_ext,
                    final_paths_dict
                )
            if os.path.isfile(cur_path):
                check_ext = self._check_ext(dirs, allowed_ext)
                if check_ext:
                    final_paths_dict[dirs] = final_paths_dict.get(dirs, cur_path)
    
    def _check_ext(self, file: str, ext: [str]):
        extracted_ext = file.split('.')[-1]
        if extracted_ext in ext:
            return True
        return False

    def _preprocess(self, text):
        text = text.lower()#converting string to lowercase
        text = re.sub(r'\d(\.|\)|-)', ' ', text)
        text = re.sub(r'\d+ words:?', ' ', text)
        res1 = re.sub(r'((http|https)://|www.).+?(\s|$)',' ',text)#removing links
        res2 = re.sub(f'[{string.punctuation}]+',' ',res1)#removing non english and special characters
        res3 = re.sub(r'[^a-z0-9A-Z\s]+',' ',res2)#removing anyother that is not consider in above
        res4 = re.sub(r'(\n)+',' ',res3)#removing all new line characters
        res4 = re.sub(r'\s[a-zA-Z]\s', ' ', res4) # removing any single char
        res = re.sub(r'\s{2,}',' ',res4)#remove all the one or more consecutive occurance of sapce
        res = res.strip()
        return res

class PatternGenTextToImageDataset(UpdatedPatternTextToImageDataset):
    def __init__(self, **kwargs):

        imsize = kwargs['size']
        root_path = kwargs['root_path']
        img_transforms = transforms.Compose([
            transforms.Resize((imsize,imsize)),
            transforms.ToTensor()
        ])
        
        csv_path = os.path.join(root_path, 'short')

        image_folder_paths = [
            os.path.join(root_path, f'dataset{i}') for i in [2,4,5,6]
        ]

        super().__init__(
            csv_path,
            image_folder_paths,
            img_transforms
        )

if __name__ == "__main__":
    # pattern_data = PrintGenerationData(size=256, root_path='/root')
    pattern_data = PatternGenTextToImageDataset(size=256, root_path='/workspace/dataset')
    input_data = pattern_data[0]
    print(input_data["image"].shape)
    print(input_data["caption"])
    print(len(pattern_data))