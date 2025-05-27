![2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/yUKVKSX2E18k0h3YwCx1h.png)

# **Imgscope-OCR-2B-0527**

> The **Imgscope-OCR-2B-0527** model is a fine-tuned version of *Qwen2-VL-2B-Instruct*, specifically optimized for *messy handwriting recognition*, *document OCR*, *realistic handwritten OCR*, and *math problem solving with LaTeX formatting*. This model is trained on custom datasets for document and handwriting OCR tasks and integrates a conversational approach with strong visual and textual understanding for multi-modal applications.

> [!warning]
Colab Demo : https://huggingface.co/prithivMLmods/Imgscope-OCR-2B-0527/blob/main/Imgscope%20OCR%202B%200527%20Demo/Imgscope-OCR-2B-0527.ipynb

---

### Key Enhancements

* **SoTA Understanding of Images of Various Resolution & Ratio**
  Imgscope-OCR-2B-0527 achieves state-of-the-art performance on visual understanding benchmarks such as MathVista, DocVQA, RealWorldQA, and MTVQA.

* **Enhanced Handwriting OCR**
  Specifically optimized for recognizing and interpreting **realistic and messy handwriting** with high accuracy. Ideal for digitizing handwritten documents and notes.

* **Document OCR Fine-Tuning**
  Fine-tuned with curated and realistic **document OCR datasets**, enabling accurate extraction of text from various structured and unstructured layouts.

* **Understanding Videos of 20+ Minutes**
  Capable of processing long videos for **video-based question answering**, **transcription**, and **content generation**.

* **Device Control Agent**
  Supports decision-making and control capabilities for integration with **mobile devices**, **robots**, and **automation systems** using visual-textual commands.

* **Multilingual OCR Support**
  In addition to English and Chinese, the model supports **OCR in multiple languages** including European languages, Japanese, Korean, Arabic, and Vietnamese.

---

### How to Use

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Imgscope-OCR-2B-0527",  # replace with updated model ID if available
    torch_dtype="auto",
    device_map="auto"
)

# Optional: Flash Attention for performance optimization
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "prithivMLmods/Imgscope-OCR-2B-0527",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# Load processor
processor = AutoProcessor.from_pretrained("prithivMLmods/Imgscope-OCR-2B-0527")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Recognize the handwriting in this image."},
        ],
    }
]

# Prepare input
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Generate output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

---

### Demo Inference

![Screenshot 2025-05-27 at 03-40-34 Gradio.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/9KiRkOGPB8cLl6VHwh2UD.png)
![Screenshot 2025-05-27 at 03-40-56 (anonymous) - output_e0fbfa20-686e-4bce-b2e8-25991be5a5a0.pdf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/VOHQIrT7hCs5afGMRROvD.png)

---

### Buffering Output (Streaming)

```python
buffer = ""
for new_text in streamer:
    buffer += new_text
    buffer = buffer.replace("<|im_end|>", "")
    yield buffer
```

---

### Key Features

1. **Realistic Messy Handwriting OCR**

   * Fine-tuned for **complex and hard-to-read handwritten inputs** using real-world handwriting datasets.

2. **Document OCR and Layout Understanding**

   * Accurately extracts text from structured documents, including scanned pages, forms, and academic papers.

3. **Image and Text Multi-modal Reasoning**

   * Combines **vision-language capabilities** for tasks like captioning, answering image-based queries, and understanding image+text prompts.

4. **Math Problem Solving and LaTeX Rendering**

   * Converts mathematical expressions and problem-solving steps into **LaTeX** format.

5. **Multi-turn Conversations**

   * Supports **dialogue-based reasoning**, retaining context for follow-up questions.

6. **Video + Image + Text-to-Text Generation**

   * Accepts inputs from videos, images, or combined media with text, and generates relevant output accordingly.

---

## **Intended Use**

**Imgscope-OCR-2B-0527** is intended for:

* Handwritten and printed document digitization
* OCR pipelines for educational institutions and businesses
* Academic and scientific content parsing, especially math-heavy documents
* Assistive tools for visually impaired users
* Robotic and mobile automation agents interpreting screen or camera data
* Multilingual OCR processing for document translation or archiving
