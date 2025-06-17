import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer
from collections import OrderedDict
import re
from collections import defaultdict

# --------------------------------------------------------------------------
# 1) CheXbert Model Definition & Loading
# --------------------------------------------------------------------------

class bert_labeler(nn.Module):
    """
    Minimal CheXbert model definition.
    """
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None):
        super(bert_labeler, self).__init__()
        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)
        hidden_size = self.bert.pooler.dense.in_features

        # 13 heads with 4 classes each => present, absent, uncertain, not mentioned
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        # 1 head for "No Finding" => 2 classes
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, input_ids, attention_mask):
        final_hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)

        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out


def load_chexbert_model_and_tokenizer(checkpoint_path, device="cpu"):
    model = bert_labeler()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if "module.bert." in key:
            new_key = key.replace("module.bert.", "bert.")
        elif "module.linear_heads." in key:
            new_key = key.replace("module.linear_heads.", "linear_heads.")
        elif "module." in key:
            new_key = key.replace("module.", "")
        else:
            new_key = key
        new_state_dict[new_key] = val

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


@torch.no_grad()
def get_chexbert_predictions(text, model, tokenizer, device="cpu"):
    text = text.strip()
    if not text:
        return None
    inputs = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    preds = []
    for logits in outputs:
        preds.append(torch.argmax(logits, dim=1).item())
    return preds

# --------------------------------------------------------------------------
# 2) Text Parsing with section_parser logic
# --------------------------------------------------------------------------

def list_rindex(lst, val):
    return len(lst) - 1 - lst[::-1].index(val)

# (Content trimmed in this simplified version: include your own section_parser logic)

# --------------------------------------------------------------------------
# 3) Token Counting
# --------------------------------------------------------------------------

def count_gpt2_tokens(text: str, tokenizer: GPT2Tokenizer) -> int:
    if not text.strip():
        return 0
    return len(tokenizer.encode(text))

# --------------------------------------------------------------------------
# 4) Main CSV creation
# --------------------------------------------------------------------------

CHEXPERT_LABEL_ORDER = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding"
]


def create_mimic_study_level_metadata_csv(
    image_root,
    text_root,
    split_csv,
    chexpert_csv,
    meta_csv,
    out_csv,
    chexbert_ckpt="/path/to/chexbert.pth",
    device="cpu"
):
    """Build and save a study-level metadata CSV with CheXbert labels."""
    chexbert_model, chexbert_tokenizer = load_chexbert_model_and_tokenizer(
        checkpoint_path=chexbert_ckpt,
        device=device
    )
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    df_split = pd.read_csv(split_csv)
    split_dict = {}
    for _, row in df_split.iterrows():
        key_3 = (int(row["subject_id"]), int(row["study_id"]), str(row["dicom_id"]))
        split_dict[key_3] = str(row["split"]).lower()

    df_chex = pd.read_csv(chexpert_csv)
    chexpert_dict_raw = {}
    for _, row in df_chex.iterrows():
        subj, stdy = int(row["subject_id"]), int(row["study_id"])
        label_data = {}
        for label in CHEXPERT_LABEL_ORDER:
            label_data[label] = row.get(label, None)
        chexpert_dict_raw[(subj, stdy)] = label_data

    def convert_labels_to_list(label_data):
        out = []
        for lbl in CHEXPERT_LABEL_ORDER:
            val = label_data.get(lbl, 0)
            if pd.isnull(val):
                val = 0
            out.append(int(val))
        return out

    df_meta = pd.read_csv(meta_csv)
    meta_dict = {}
    for _, row in df_meta.iterrows():
        key_3 = (int(row["subject_id"]), int(row["study_id"]), str(row["dicom_id"]))
        meta_dict[key_3] = {
            "ViewPosition": row.get("ViewPosition", None),
            "PerformedProcedureStepDescription": row.get("PerformedProcedureStepDescription", None)
        }

    study_dict = {}

    for subdir, _, files in os.walk(image_root):
        jpgs = [f for f in files if f.lower().endswith(".jpg")]
        if not jpgs:
            continue

        rel_subdir = os.path.relpath(subdir, image_root)
        parts = rel_subdir.split(os.sep)
        if len(parts) < 3:
            continue

        patient_folder = parts[-2]
        study_folder = parts[-1]
        if patient_folder.startswith("p"):
            patient_id = int(patient_folder[1:])
        else:
            patient_id = int(patient_folder)
        if study_folder.startswith("s"):
            study_id = int(study_folder[1:])
        else:
            study_id = int(study_folder)

        text_file = os.path.join(text_root, rel_subdir + ".txt")
        with open(text_file, 'r', encoding='utf-8') as f:
            all_text = f.read()
        findings_text = ""
        impression_text = ""
        if "FINDINGS:" in all_text:
            split = all_text.split("FINDINGS:")
            if len(split) > 1:
                findings_text = split[1].split("IMPRESSION:")[0].strip()
        if "IMPRESSION:" in all_text:
            impression_text = all_text.split("IMPRESSION:")[-1].strip()

        chexbert_text = impression_text if impression_text else findings_text
        chexbert_pred = get_chexbert_predictions(
            chexbert_text, chexbert_model, chexbert_tokenizer, device=device
        )
        if chexbert_pred is None:
            chexbert_pred = [3]*13 + [0]

        raw_labels = chexpert_dict_raw.get((patient_id, study_id), {})
        chex_labels_list = convert_labels_to_list(raw_labels)
        discrepancy = 0
        for i in range(14):
            chexpert_is_present = (chex_labels_list[i] == 1)
            chexbert_is_present = (chexbert_pred[i] == 1)
            if chexpert_is_present != chexbert_is_present:
                discrepancy = 1
                break

        findings_tokens = count_gpt2_tokens(findings_text, gpt2_tokenizer)
        impression_tokens = count_gpt2_tokens(impression_text, gpt2_tokenizer)
        all_text_tokens = count_gpt2_tokens(all_text, gpt2_tokenizer)
        image_paths_joined = ";".join([os.path.join(subdir, f) for f in jpgs])
        chexpert_labels_str = ";".join(str(x) for x in chex_labels_list)
        chexbert_labels_str = ";".join(str(x) for x in chexbert_pred)

        study_dict[(patient_id, study_id)] = {
            "subject_id": patient_id,
            "study_id": study_id,
            "split": split_dict[(patient_id, study_id, jpgs[0].split('.')[0])],
            "image_paths": image_paths_joined,
            "findings": findings_text,
            "impression": impression_text,
            "report_all": all_text,
            "has_findings": 1 if findings_text.strip() else 0,
            "has_impression": 1 if impression_text.strip() else 0,
            "findings_tokens_gpt2": findings_tokens,
            "impression_tokens_gpt2": impression_tokens,
            "all_text_tokens_gpt2": all_text_tokens,
            "chexpert_labels": chexpert_labels_str,
            "chexbert_labels": chexbert_labels_str,
            "chex_label_diff": discrepancy,
            "num_images": len(jpgs)
        }

    df_out = pd.DataFrame(list(study_dict.values()))
    df_out.to_csv(out_csv, index=False)
    print(f"Created {len(df_out)} study-level records in {out_csv}.")


if __name__ == "__main__":
    create_mimic_study_level_metadata_csv(
        image_root="/example/mimic-cxr-jpg/files",
        text_root="/example/mimic-cxr-reports/files",
        split_csv="/example/mimic-cxr-2.0.0-split.csv",
        chexpert_csv="/example/mimic-cxr-2.0.0-chexpert.csv",
        meta_csv="/example/mimic-cxr-2.0.0-metadata.csv",
        out_csv="/example/mimic_metadata_bert_base_macbook.csv",
        chexbert_ckpt="/example/chexbert.pth",
        device="cpu"
    )
