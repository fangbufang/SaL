# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import math

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer
from mmf.utils.build import build_image_encoder
from omegaconf import OmegaConf
from torch import nn
from packages.transformers import T5Tokenizer, T5ForConditionalGeneration,T5Config
from packages.transformers.models.t5.modeling_t5 import T5LayerNorm



logger = logging.getLogger(__name__)


@registry.register_model("sal")
class SaL(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = T5Config(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/sal/defaults.yaml"

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
       

  

    def _build_txt_encoding(self):
        self.qestion_tokenizer = T5Tokenizer.from_pretrained("/data/fcy/sal/t5-base")
       
    def _build_obj_encoding(self):
     
        self.obj_vit_fc7 = nn.Linear(2048,768)
        self.obj_bbox = nn.Linear(4,768)
        # self.ocr_token_index_embeddings = nn.Embedding(512,768)
        self.finetune_modules.append(
            {"module": self.obj_vit_fc7, "lr_scale": self.config.lr_scale_frcn}
        )
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)
        self.obj_feat_layer_norm = T5LayerNorm(self.mmt_config.hidden_size)#change
        self.obj_bbox_layer_norm = T5LayerNorm(self.mmt_config.hidden_size)#change
    def _build_ocr_encoding(self):
     
        self.ocr_vit_fc7 = nn.Linear(2048,768)
        self.ocr_bbox = nn.Linear(4,768)
        # self.ocr_circle = nn.Linear(700,768)
        # self.ocr_token_index_embeddings = nn.Embedding(512,768)
        self.finetune_modules.append(
            {"module": self.ocr_vit_fc7, "lr_scale": self.config.lr_scale_frcn}
        )
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)
        self.ocr_feat_layer_norm = T5LayerNorm(self.mmt_config.hidden_size)#change
        self.ocr_bbox_layer_norm = T5LayerNorm(self.mmt_config.hidden_size)#change


    def _build_mmt(self):
        self.latr = T5ForConditionalGeneration.from_pretrained("/data/fcy/sal/t5-base").to(self.device)
        self.type_embed = nn.Embedding(3,768)
        self.embedding = self.latr.get_input_embeddings()
        # self.txt_position = nn.Embedding(50,768,device=self.device)
        
        

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append(
            {"module": self.latr, "lr_scale": self.config.lr_scale_mmt}
        )

   

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_mmt_and_output(sample_list, fwd_results)

        # only keep scores in the forward pass results
        results = {"scores": fwd_results["scores"],"scores2": fwd_results["scores2"]}
        return results

    def _forward_txt_encoding(self, sample_list, fwd_results):
        question = self.qestion_tokenizer(
             [sequence for sequence in sample_list.text],
                # padding="longest",
                padding=True,
                truncation=True,
                max_length=45,
                return_tensors="pt",).to(self.device)

        # binary mask of valid text (question words) vs padding
        # = pad_sequence([question.attention_mask,self.img_pad])[:,0]
        # = pad_sequence([img_fea_0,self.img_pad])[:,0]
        batch,legth_q = question.attention_mask.size()
        pad = torch.zeros([batch,45-legth_q],device=self.device,dtype=torch.long)
        fwd_results["txt_mask"] = torch.cat([question.attention_mask,pad],dim=1)
        fwd_results["txt_input_ids"] =  torch.cat([question.input_ids,pad],dim=1)
    def _forward_obj_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)

        
        fwd_results["obj_mask"] =sample_list.obj_text_index_mask
        fwd_results["obj_input_ids"] = sample_list.obj_text_index
        obj_bbox = sample_list.obj_bbox_coordinates
        obj_bbox_embeds = self.obj_bbox(obj_bbox)
       
        obj_vit = sample_list.image_feature_0 
        obj_fc7 = self.obj_vit_fc7(obj_vit)
        obj_fc7 = self.obj_feat_layer_norm(obj_fc7)
        fwd_results["obj_vit_featurte"]=obj_fc7
        fwd_results["obj_bbox_embeds"] =  self.obj_bbox_layer_norm(obj_bbox_embeds)

    def _forward_ocr_encoding(self, sample_list, fwd_results):

        
        fwd_results["ocr_mask"] = sample_list.ocr_text_index_mask
        fwd_results["ocr_input_ids"] = sample_list.ocr_text_index
        ocr_bbox = sample_list.ocr_bbox_coordinates
        ocr_bbox_embeds = self.ocr_bbox(ocr_bbox)
      
        ocr_vit = sample_list.image_feature_1 #  ocr_vit is frcnn feature !
        ocr_fc7 = self.ocr_vit_fc7(ocr_vit)
        ocr_fc7 =self.ocr_feat_layer_norm(ocr_fc7)
        fwd_results["ocr_vit_featurte"]=ocr_fc7
        fwd_results["ocr_bbox_embeds"] =  self.ocr_bbox_layer_norm(ocr_bbox_embeds)
       
   

        

    def _forward_mmt(self, sample_list, fwd_results):
     
      
        batch_size ,txt_length= fwd_results["txt_mask"].size()
        ocr_length = fwd_results["ocr_mask"].size(1)
        q_type = torch.zeros(45,dtype=torch.long,device=fwd_results["ocr_mask"].device) 
        ocr_type = torch.zeros(350,dtype=torch.long,device=fwd_results["ocr_mask"].device) + 1
        obj_type = torch.zeros(250,dtype=torch.long,device=fwd_results["ocr_mask"].device) +2
        q_type_embed = self.type_embed(q_type ) 
        ocr_type_embed = self.type_embed(ocr_type )
        obj_type_embed = self.type_embed(obj_type )
        latr_input_ids = torch.cat([fwd_results["txt_input_ids"],fwd_results["ocr_input_ids"],fwd_results["obj_input_ids"]],dim=1)
        latr_ocr_ques_embeds = self.embedding(latr_input_ids)
      
        ocr_vit = fwd_results["ocr_vit_featurte"]
        obj_vit = fwd_results["obj_vit_featurte"]
      
        latr_input_embeds = latr_ocr_ques_embeds
        obj_bbox_token_index =self.obj_drop(fwd_results["obj_bbox_embeds"]+obj_vit)
        ocr_bbox_token_index =self.ocr_drop(fwd_results["ocr_bbox_embeds"]+ocr_vit)
     
        latr_input_embeds[:,:txt_length] +=  q_type_embed
        latr_input_embeds[:,txt_length:txt_length+ocr_length] += ocr_type_embed
        latr_input_embeds[:,txt_length+ocr_length:] +=  obj_type_embed

        latr_input_embeds[:,txt_length:txt_length+ocr_length] += ocr_bbox_token_index
        latr_input_embeds[:,txt_length+ocr_length:] +=  obj_bbox_token_index
      
   
        latr_mask = torch.cat([fwd_results["txt_mask"],fwd_results["ocr_mask"],fwd_results["obj_mask"]],dim=1)
        labels = sample_list.train_prev_inds
      
      
        ocr_circle_dist = sample_list.ocr_circle_dist
        
        # SCP module modified in Line 551  packages\transformers\models\t5\modeling_t5.py
        latr_output = self.latr(inputs_embeds=latr_input_embeds,labels=labels,ocr_circle_dist=ocr_circle_dist,attention_mask=latr_mask)
        scores = latr_output.logits
        fwd_results["scores"] = scores.argmax(dim=-1)
        fwd_results["scores2"] = scores

    def _forward_mmt_generate(self, sample_list, fwd_results):
        # first forward the text BERT layers
        
        batch_size ,txt_length= fwd_results["txt_mask"].size()
        ocr_length = fwd_results["ocr_mask"].size(1)
        q_type = torch.zeros(45,dtype=torch.long,device=fwd_results["ocr_mask"].device) 
        ocr_type = torch.zeros(350,dtype=torch.long,device=fwd_results["ocr_mask"].device) + 1
        obj_type = torch.zeros(250,dtype=torch.long,device=fwd_results["ocr_mask"].device) +2
        q_type_embed = self.type_embed(q_type ) 
        ocr_type_embed = self.type_embed(ocr_type )
        obj_type_embed = self.type_embed(obj_type )
      
        latr_input_ids = torch.cat([fwd_results["txt_input_ids"],fwd_results["ocr_input_ids"],fwd_results["obj_input_ids"]],dim=1)
        latr_ocr_ques_embeds = self.embedding(latr_input_ids)
      
        ocr_vit = fwd_results["ocr_vit_featurte"]
        obj_vit = fwd_results["obj_vit_featurte"]
    
        latr_input_embeds = latr_ocr_ques_embeds
        obj_bbox_token_index =self.obj_drop(fwd_results["obj_bbox_embeds"]+obj_vit)
        ocr_bbox_token_index =self.ocr_drop(fwd_results["ocr_bbox_embeds"]+ocr_vit)
     
        latr_input_embeds[:,:txt_length] +=  q_type_embed
        latr_input_embeds[:,txt_length:txt_length+ocr_length] += ocr_type_embed
        latr_input_embeds[:,txt_length+ocr_length:] +=  obj_type_embed

        latr_input_embeds[:,txt_length:txt_length+ocr_length] += ocr_bbox_token_index
        latr_input_embeds[:,txt_length+ocr_length:] +=  obj_bbox_token_index
      
       
        latr_mask = torch.cat([fwd_results["txt_mask"],fwd_results["ocr_mask"],fwd_results["obj_mask"]],dim=1)
        labels = sample_list.train_prev_inds
      

        ocr_circle_dist = sample_list.ocr_circle_dist
        
        # ocr_two = self.ocr_circle_norm(self.ocr_circle(sample_list.ocr_two))
        latr_output = self.latr.generate(inputs_embeds = latr_input_embeds,max_new_tokens=25,ocr_circle_dist=ocr_circle_dist,attention_mask=latr_mask,do_sample=False,)
      
        scores = torch.zeros((latr_output.size(0),26),dtype=latr_output.dtype).to(latr_output.device)
        scores[:,:latr_output.size(1)] = latr_output
        fwd_results["scores"] = scores
        fwd_results["scores2"] = scores
      


    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
           
            self._forward_mmt(sample_list, fwd_results)
            
        else:
            
           

            # greedy decoding at test time
            
            self._forward_mmt_generate(sample_list, fwd_results)
                


    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

    @classmethod
    def update_registry_for_pretrained(cls, config, checkpoint, full_output):
        from omegaconf import OmegaConf

        # Hack datasets using OmegaConf
        datasets = full_output["full_config"].datasets
        dataset = datasets.split(",")[0]
        config_mock = OmegaConf.create({"datasets": datasets})
        registry.register("config", config_mock)
        registry.register(
            f"{dataset}_num_final_outputs",
            # Need to add as it is subtracted
            checkpoint["classifier.module.weight"].size(0)
            + config.classifier.ocr_max_num,
        )
        # Fix this later, when processor pipeline is available
        answer_processor = OmegaConf.create({"BOS_IDX": 1})
        registry.register(f"{dataset}_answer_processor", answer_processor)







