import base64
import os


def _get_vertex_runtime():
    project = os.environ.get("VERTEXAI_PROJECT")
    location = os.environ.get("VERTEXAI_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("VERTEXAI_PROJECT is not set")

    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    import vertexai.preview.generative_models as generative_models

    vertexai.init(project=project, location=location)
    return GenerativeModel, Part, generative_models


def _get_zhipu_client():
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        raise RuntimeError("ZAI_API_KEY is not set")

    from zai import ZhipuAiClient

    return ZhipuAiClient(api_key=api_key)


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        return f"Image file does not exist: {image_path}"
    except Exception as exc:
        return f"Failed to encode image: {exc}"


def evaluate(image_urls):
    prompt = """
**Act as a professional image quality and identity evaluation system. You will receive one reference image (the first one) followed by multiple generated images (others) for assessment. For each generated image, evaluate based on these criteria:**

1.  **Structural Integrity and Reasonableness (40% weight):** Assess the inherent rationality of the generated image itself. For human/animal faces: evaluate facial symmetry, proportional distribution of facial features, anatomical correctness, and natural appearance. For objects: evaluate structural coherence, physical plausibility, and absence of deformities or artifacts.

2.  **Identity Faithfulness to Reference (60% weight):** Determine the degree to which the person/object in the generated image is the same as in the reference image. Consider facial features, distinctive characteristics, and overall likeness for persons; consider form, texture, and defining attributes for objects.

**Scoring Guidelines:**
- Provide a single score from 1 to 100 for each generated image, where a higher score indicates a better quality image that is more faithful to the reference.
- **Ensure meaningful score distribution:** Apply strict grading with significant variance (e.g., 50-100 range) to clearly differentiate between excellent, good, average, and poor results. Avoid score compression. Please ensure the average score is 80.
- Output **only** a Python list of numerical scores (e.g., `[85, 72, 78, 95, 70]`) with no additional text, explanations, or formatting.
"""
    GenerativeModel, Part, generative_models = _get_vertex_runtime()
    model = GenerativeModel("gemini-2.5-pro")

    contents = []
    for filename in image_urls:
        with open(filename, "rb") as file:
            image_content = file.read()
        contents.append(Part.from_data(image_content, mime_type="image/png"))
    contents.append(prompt)

    generation_config = {"max_output_tokens": 2048, "temperature": 1e-5, "top_p": 1.0}
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    response = model.generate_content(contents, generation_config=generation_config, safety_settings=safety_settings)
    return response.text


def glm_evaluate(image_paths):
    client = _get_zhipu_client()
    prompt = """
**Act as a professional image quality and identity evaluation system. You will receive one reference image (the first one) followed by multiple generated images (others) for assessment. For each generated image, evaluate based on these criteria:**

1.  **Structural Integrity and Reasonableness (40% weight):** Assess the inherent rationality of the generated image itself. For human/animal faces: evaluate facial symmetry, proportional distribution of facial features, anatomical correctness, and natural appearance. For objects: evaluate structural coherence, physical plausibility, and absence of deformities or artifacts.

2.  **Identity Faithfulness to Reference (60% weight):** Determine the degree to which the person/object in the generated image is the same as in the reference image. Consider facial features, distinctive characteristics, and overall likeness for persons; consider form, texture, and defining attributes for objects.

**Scoring Guidelines:**
- Provide a single score from 1 to 100 for each generated image, where a higher score indicates a better quality image that is more faithful to the reference.
- **Ensure meaningful score distribution:** Apply strict grading with significant variance (e.g., 50-100 range) to clearly differentiate between excellent, good, average, and poor results. Avoid score compression. Please ensure the average score is 80.
- Output **only** a Python list of numerical scores (e.g., `[85, 72, 78, 95, 70]`) with no additional text, explanations, or formatting.
"""

    content = [
        {
            "type": "image_url",
            "image_url": {"url": image_to_base64(path)},
        }
        for path in image_paths
    ]
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model="glm-4.1v-thinking-flash",
        stream=False,
        thinking={"type": "enabled"},
        do_sample=False,
        temperature=0,
        top_p=0.9,
        messages=[
            {"role": "system", "content": "You are a picture scorer."},
            {"role": "user", "content": content},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


def extract(text, cla=None):
    prompt = f'''
    Please extract the information about the {cla} described in the text: {text}.
    Output them in a simple sentence less than 20 words.
    If there is no information that describes {cla}, then give an empty output.
    Notice: Do not output any other information.
    '''
    GenerativeModel, Part, generative_models = _get_vertex_runtime()
    model = GenerativeModel("gemini-2.5-pro")
    generation_config = {"max_output_tokens": 2048, "temperature": 1e-5, "top_p": 1.0}
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    response = model.generate_content([prompt], generation_config=generation_config, safety_settings=safety_settings)
    return response.text


def glm_extract(text, cla=None):
    client = _get_zhipu_client()
    prompt = f'''
    Please extract the information about the {cla} described in the text: {text}.
    Output them in a simple sentence less than 20 words.
    If there is no information that describes {cla}, then give an empty output.
    Notice: Do not output any other information.
    '''
    response = client.chat.completions.create(
        model="glm-4.1v-thinking-flash",
        stream=False,
        thinking={"type": "enabled"},
        do_sample=False,
        temperature=0.1,
        top_p=0.95,
        messages=[
            {"role": "system", "content": "You are a text analyst."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    raise SystemExit("Configure environment variables and call these helpers from training code.")
