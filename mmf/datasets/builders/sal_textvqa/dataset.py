# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.utils.text import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from packages.transformers import T5Tokenizer
# from dict_trie import Trie
from pyxdameraulevenshtein import  normalized_damerau_levenshtein_distance_seqs
import random
from collections import defaultdict
import math

class SalDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("sal_textvqa", config, dataset_type, index=imdb_file_index)
        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info
        self.contex_obj_ocr_pad = torch.zeros((3,2048))
        self.tokenizer = T5Tokenizer.from_pretrained("/data/fcy/sal/t5-base")
        import numpy as np

        self.x_size, y_size = 11, 11
        x_arr, y_arr = np.mgrid[0:self.x_size, 0:y_size]
        # used for SCP module
        for x in range(self.x_size):
            for y in range(y_size):
                cell = (x,y)
                a = (y_arr - cell[1])/(x_arr - cell[0])
                a[np.isnan(a)]=0
                dists = np.sqrt((x_arr - cell[0])**2 + (y_arr - cell[1])**2).reshape(1,self.x_size,y_size) #* left_to_right
                if y==0:
                    dists_align_y = dists
                 
                else:
                    dists_align_y = np.concatenate((dists_align_y,dists),axis=0)
                 
            dists_align_y = dists_align_y.reshape(1,y_size,self.x_size,y_size)
         
            if x == 0:
                dists_align_all = dists_align_y
             
            else:
                dists_align_all = np.concatenate((dists_align_all,dists_align_y),axis=0)
              
        self.dists_align_all = dists_align_all *5



     


    def get_bbox_area(self, pos):

        height = pos[3] - pos[1]
        width = pos[2] - pos[0]
        area = height * width
        return area
    def preprocess_sample_info(self, sample_info):
        path = self._get_path_based_on_index(self.config, "annotations", self._index)
        # NOTE, TODO: Code duplication w.r.t to STVQA, revisit
        # during dataset refactor to support variable dataset classes
        if "stvqa" in path:
            feature_path = sample_info["feature_path"]
            append = "train"

            if self.dataset_type == "test":
                append = "test_task3"

            if not feature_path.startswith(append):
                feature_path = append + "/" + feature_path

            sample_info["feature_path"] = feature_path
            return sample_info
        # COCO Annotation DBs have corrext feature_path
        elif "COCO" not in sample_info["feature_path"]:
            sample_info["feature_path"] = sample_info["image_path"].replace(
                ".jpg", ".npy"
            )
        return sample_info

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_prediction(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()
        answer_words=self.tokenizer.batch_decode(pred_answers, skip_special_tokens=True)
        image_ids = report.image_id.cpu().numpy()
        # context_tokens = report.context_tokens.cpu().numpy()
        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            image_id = byte_tensor_to_object(image_ids[idx])
            # tokens = byte_tensor_to_object(context_tokens[idx])
            
            pred_source = []
            answer_word=answer_words[idx]
            pred_source.append("VOCAB")
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = answer_word.replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "image_id": image_id,
                "answer": pred_answer,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]
        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)
        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead
        if self.config.pretrain:
            current_sample.pop('image_feature_0')
            current_sample.pop('image_feature_1')
            current_sample.pop('obj_bbox_coordinates')
            current_sample.pop('ocr_bbox_coordinates')
       
        if hasattr(current_sample, "image_info_0"):
            for k in list(current_sample.image_info_0):
                if k != "max_features":
                    current_sample.image_info_0.pop(k)
        if hasattr(current_sample, "image_info_1"):
            for k in list(current_sample.image_info_1):
                if k != "max_features":
                    current_sample.image_info_1.pop(k)

        return current_sample

    def add_sample_details(self, sample_info, sample):
        sample.image_id = object_to_byte_tensor(sample.image_id)

        # 1. Load text (question words)
        question_str = (
            sample_info["question"]
            if "question" in sample_info
            else sample_info["question_str"]
        )
        sample.text = "question: "+question_str.lower()  
        if self.config.pretrain:
            sample.text = "question: "+question_str.lower() +" "+ sample_info["valid_answers"][random.randint(0,len(sample_info['valid_answers'])-1)]


        # 2. Load object
        # object bounding box information


        if "obj_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):

            if sample.image_info_0.bbox.ndim == 1:
                sample.image_info_0.bbox = np.array([[0,0,0,0,0,0]],dtype=np.float32)
            sample.image_info_0.bbox =  np.concatenate((np.zeros((3,6)),sample.image_info_0.bbox[sample.image_info_0.obj_token_lengths]),axis=0,dtype=np.float32)
            sample.obj_bbox_coordinates = self.copy_processor(
                {"blob": sample.image_info_0.bbox}
            )["blob"][:250,:4]

        sample.obj_bbox_coordinates[:,0]=(sample.obj_bbox_coordinates[:,0]/sample_info["image_width"]).clamp(min=0,max=0.999)
        sample.obj_bbox_coordinates[:,2]=(sample.obj_bbox_coordinates[:,2]/sample_info["image_width"]).clamp(min=0,max=0.999)
        sample.obj_bbox_coordinates[:,1]=(sample.obj_bbox_coordinates[:,1]/sample_info["image_height"]).clamp(min=0,max=0.999)
        sample.obj_bbox_coordinates[:,3]=(sample.obj_bbox_coordinates[:,3]/sample_info["image_height"]).clamp(min=0,max=0.999)

        
        # 3. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info["ocr_tokens"] = []
            sample_info["ocr_info"] = []
            if "ocr_normalized_boxes" in sample_info:
                sample_info["ocr_normalized_boxes"] = np.zeros((0, 4), np.float32)
            # clear OCR visual features
            if "image_feature_1" in sample:
                sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
            return sample

        ocr_ans_sorted_tokens = np.array(sample_info["ocr_t5_token"],dtype=np.int64)
        ocr_ans_sorted_tokens_index = np.array(sample_info["ocr_tokens_index"],dtype=np.int64)



        if "ocr_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):
            # New imdb format: OCR bounding boxes are already pre-computed
            max_len = self.config.processors.answer_processor.params.max_length
            if sample_info["wordlevel_ocr_normalized_boxes"].ndim == 1:
                sample_info["wordlevel_ocr_normalized_boxes"] = np.array([[0,0,0,0,0,0]],dtype=np.float32)
                

            #! for Rosetta OCR
            ocr_normalized_boxes = np.concatenate((np.zeros((3,4)),sample_info["wordlevel_ocr_normalized_boxes"][ocr_ans_sorted_tokens_index][:,:4]),axis=0,dtype=np.float32)
            sample.ocr_bbox_coordinates = self.copy_processor(
                {"blob": ocr_normalized_boxes}
            )["blob"][:350,:4]
            sample.ocr_bbox_coordinates=sample.ocr_bbox_coordinates
        elif self.use_ocr_info and "ocr_info" in sample_info:
            # Old imdb format: OCR bounding boxes are computed on-the-fly
            # from ocr_info
            sample.ocr_bbox_coordinates = self.bbox_processor(
                {"info": sample_info["ocr_info"]}
            )["bbox"].coordinates
        
       

        # used for SCP module
        if not self.config.pretrain:
            x_ocr_c, y_ocr_c = (sample.ocr_bbox_coordinates[:,0] + sample.ocr_bbox_coordinates[:,2])/2,\
            (sample.ocr_bbox_coordinates[:,1] + sample.ocr_bbox_coordinates[:,3])/2
            x_ocr_c = x_ocr_c.clamp(min=0,max=0.999)
            y_ocr_c = y_ocr_c.clamp(min=0,max=0.999)
            ocr_c_x_index = np.int32(np.floor(x_ocr_c*self.x_size))
            ocr_c_y_index = np.int32(np.floor(y_ocr_c*self.x_size))
            ocr_circle_dist = self.dists_align_all[ocr_c_x_index,ocr_c_y_index][:,ocr_c_x_index,ocr_c_y_index]
            sample.ocr_circle_dist = torch.tensor(ocr_circle_dist).long()
      


        
        sample.image_feature_1[3:3+len(ocr_ans_sorted_tokens_index)] = sample.image_feature_1[ ocr_ans_sorted_tokens_index][:512-3]
        sample.image_feature_1[:3]=self.contex_obj_ocr_pad
        sample.image_feature_1=sample.image_feature_1[:350]
        sample.image_feature_0[3:3+len(sample.image_info_0.obj_token_lengths)] = sample.image_feature_0[ sample.image_info_0.obj_token_lengths][:512-3]
        sample.image_feature_0[:3]=self.contex_obj_ocr_pad
        sample.image_feature_0= sample.image_feature_0[:250]

        # process t5 tokens in annotation files e.g ./data/textvqa/defaults/annotations/train_amazon_ocr.npy split OCR texts in a image
        # then replace the index 1 with 32099 or other tokens in vocabulary
        text_index = torch.zeros(350,dtype=torch.long)
        # OCR:
        text_index[:3]=torch.tensor([2625,10,1])
        text_index[3:3+len(ocr_ans_sorted_tokens)]=torch.tensor(ocr_ans_sorted_tokens)[:350-3]
        text_index = torch.where(text_index==1,32099,text_index)
        sample.ocr_text_index = text_index
        sample.ocr_text_index_mask = (text_index!=0).long()
        sample.ocr_geo_mask = (text_index!=32099).long()

        obj_text_index = torch.zeros(250,dtype=torch.long)
        # OBJ:
        obj_text_index[:3]=torch.tensor([4820, 10, 1])
        obj_text_index[3:3+len(sample.image_info_0.obj_t5_token)]=torch.tensor(sample.image_info_0.obj_t5_token)[:250-3]
        obj_text_index = torch.where(obj_text_index==1,32099,obj_text_index)
        sample.obj_text_index = obj_text_index
        sample.obj_text_index_mask = (obj_text_index!=0).long()


        return sample

    def add_answer_info(self, sample_info, sample):
        # Load real answers from sample_info
        answers = sample_info.get("answers", [])
        answer_processor_arg = {"answers": answers}
        if sample.generate_flag == 1 and self._dataset_type=='train':
            answers = sample.get("generate_ans", [])
            answer_processor_arg = {"answers": answers}

        answer_processor_arg["tokens"] = sample.pop("ocr_tokens", [])

        processed_answers = self.answer_processor(answer_processor_arg)

        assert not self.config.fast_read, (
            "In TextVQADataset, online OCR sampling is incompatible "
            "with fast_read, so fast_read is currently not supported."
        )

        sample.update(processed_answers)
        sample.answers = object_to_byte_tensor(answers)

        if "answers_scores" in sample:
            sample.targets = sample.pop("answers_scores")

        return sample


