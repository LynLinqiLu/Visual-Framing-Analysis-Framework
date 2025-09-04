import os
import csv
import base64
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
import random
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Debug flag - set to False to disable tqdm
USE_TQDM = True

random.seed(42)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze protest images using various AI models')
    
    # API and model selection
    parser.add_argument('--api', type=str, default='ollama',
                      choices=['ollama', 'openai', 'vllm'],
                      help='API type to use (default: ollama)')
    parser.add_argument('--model', type=str, default='gemma3:27b-it-q8_0',
                      help='Model to use (default: gemma3:27b-it-q8_0)')
    
    # File paths
    parser.add_argument('--image-folder', type=str,
                      default='./images',
                      help='Path to folder containing images to analyze')
    parser.add_argument('--output-csv', type=str,
                      default='protest_analysis.csv',
                      help='Path to output CSV file')
    
    # vLLM specific
    parser.add_argument('--vllm-url', type=str, default='http://localhost:8001',
                      help='URL of vLLM server (default: http://localhost:8001)')
    
    # Processing options
    parser.add_argument('--no-progress-bar', action='store_true',
                      help='Disable progress bar')
    parser.add_argument('--sample-size', type=int, default=200,
                      help='Number of images to randomly sample (default: 200)')
    
    return parser.parse_args()

# Model configurations
MODEL_CONFIGS = {
    "gemma3:27b-it-q8_0": {
        "temperature": 0.1,
        "num_ctx": 8192,
        "num_predict": 2000
    },
    "qwen2.5vl:72b-q4_K_M": {
        "temperature": 0.1,
        "num_ctx": 8192,
        "num_predict": 2000
    },
    "llava:13b": {
        "temperature": 0.1,
        "num_ctx": 8192,
        "num_predict": 2000
    },
    # vLLM models
    "OpenGVLab/InternVL3-14B": {
        "temperature": 0.1,
        "max_tokens": 2000
    },
    "OpenGVLab/InternVL3-38B": {
        "temperature": 0.1,
        "max_tokens": 2000
    }
}

# API configurations
API_CONFIGS = {
    "ollama": {
        "gemma3:27b-it-q8_0": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": 2000
        },
        "qwen2.5vl:72b-q4_K_M": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": 2000
        },
        "llava:13b": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": 2000
        }
    },
    "openai": {
        "gpt-4-vision-preview": {
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "gpt-4o": {
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "vllm": {
        "OpenGVLab/InternVL3-14B": {
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "OpenGVLab/InternVL3-38B": {
            "temperature": 0.1,
            "max_tokens": 2000
        }
    }
}

# Analysis prompt and background knowledge
previous_background_knowledge = """You are an expert in visual content analysis, specializing in interpreting images related to social movements and protests. Your task is to analyze images to identify specific visual framing depicting 'Conflict', 'Peace', 'Protester Solidarity', and 'Police Solidarity' and to list all relevant objects, people, signs, or elements in the image that contribute to these framings.

**Definitions:**

- **Conflict Frame**: An image of a protest where people are engaged in a tense, confrontational situation. This includes scenes of conflict between protesters and police (such as police officers in riot gear facing angry protesters), conflict between different groups of protesters (such as opposing factions clashing or fighting), or protesters engaging in property destruction. Visual elements may include: throwing objects or wielding sticks, shouting and making aggressive gestures, indications of property damage like broken windows, vandalized buildings, burning/burned vehicles, fire, or smoke. The overall atmosphere conveys confrontation, aggression, physical conflict, and clear divisions between opposing groups, whether police vs protesters or between protester factions.

- **Peace Frame**: An image of a protest characterized by calm, non-violent, non-confrontational interactions and peaceful assembly. This includes scenes where protesters are engaged in orderly demonstrations, peaceful vigils, or constructive dialogue. Visual elements may include: people sitting or standing quietly, holding candles, praying together, or engaging in organized peaceful activities; protesters and police without tactical formations or weapons displayed; calm facial expressions and relaxed body language from all parties; absence of aggressive gestures, weapons, or property damage. The overall atmosphere conveys non-violent expression of views, with no visible tension.

- **Protester Solidarity Frame**: An image showing protesters united in common purpose through collective action or mutual support between protesters. Visual elements may include: groups moving or standing as one cohesive unit; physical connections like linked arms, holding hands, or group embraces; synchronized actions such as collective kneeling or raised fists in unison; shared symbols including matching clothing or unified signs; inclusive participation across diverse ages and backgrounds; supportive gestures like helping others march or sharing supplies. Protesters actively protect and defend each other, such as forming human chains to shield vulnerable members from police, pulling injured protesters to safety, or intervening when fellow protesters face police aggression. Emotional indicators may show joy, determination, hope, or solemn resolve, with protesters comforting one another or engaging in celebratory elements like singing together. The scene may include community care, creative expression, memorial elements, family presence, and collective resistance, with protesters showing solidarity both in assembly and in defending their shared right to protest.

- **Police Solidarity Frame**: An image showing police displaying unity implying connections or mutual support specifically between officers. Visual elements may include: officers helping or protecting fellow officers; providing physical or emotional support to colleagues; officers working in coordinated pairs or small groups to assist each other; sharing equipment or resources between officers; displaying concern for fellow officers' wellbeing. Emotional indicators may show camaraderie, trust, and care between officers. The scene shows police-to-police supportive interactions like officers checking on each other's safety, providing backup, or offering encouragement to colleagues. This focuses solely on solidarity between law enforcement personnel, not their interactions with protesters or the public.

**Key Definitions:**

- **Protester**: Any person who appears to be participating in the demonstration, rally, or protest. This includes people holding signs, chanting, marching, wearing protest-related clothing or symbols, or otherwise engaged in protest activities. Does not include bystanders, journalists, or law enforcement.

- **police_regular_gear**: Standard police uniform including regular duty belt with holstered weapon, radio, handcuffs, and standard police hat or cap. Does not include tactical helmets, riot shields, gas masks, or military-style equipment.

- **police_riot_gear**: Tactical equipment including helmets, shields, batons, gas masks, body armor, riot control weapons, and other military-style protective or crowd control equipment.

- **standing_confrontation**: A tense face-to-face standoff specifically between protesters and police, where both groups are positioned in opposition to each other, typically with little physical distance separating them.

- **peaceful_gathering**: Protesters assembled in a calm, non-confrontational manner without signs of aggression, violence, or tension. May include sitting, standing quietly, or engaging in organized peaceful activities.

- **vigils**: Solemn gatherings where people hold candles, observe moments of silence, or engage in quiet commemorative activities, often in memory of someone or to mark a significant event.

- **comforting_or_hugging**: Physical gestures of support and care between individuals, including embracing, consoling someone who is upset, providing emotional support, or other nurturing physical contact.

- **Happy (Protester Emotions)**: Visible signs of joy, positivity, or celebration including smiling, laughing, cheerful expressions, or celebratory gestures.

- **vehicle_none**: There are no vehicles of any type visible in the image.

- **vehicle_unclear**: There are vehicles visible in the image, but their type (civilian, police, emergency) cannot be clearly determined due to distance, angle, obstruction, or image quality.

- **crowd_dynamics_chaotic_or_disorganized**: The crowd appears scattered, moving in different directions, lacks clear formation or structure, with people acting independently rather than as a cohesive group.

- **crowd_dynamics_organized_or_unified**: The crowd appears coordinated, moving together in formation, following clear patterns or leadership, with people acting collectively as a unified group.

- **action_fighting**: Physical combat between individuals, including punching, kicking, wrestling, or other forms of direct physical violence between people.

- **protester_violence**: Any form of aggressive or violent behavior exhibited by protesters, including but not limited to fighting, throwing objects, damaging property, or threatening gestures toward others.

- **police_using_force**: Any instance where law enforcement officers employ physical force, weapons, restraints, or coercive tactics against protesters or other individuals.

- **aggressive_gestures**: Threatening or hostile body language including clenched fists raised in anger, pointing aggressively, shoving motions, or other intimidating physical postures that convey hostility or intent to harm.

- **projectiles**: Objects that have been thrown or launched through the air, including rocks, bottles, tear gas canisters, or any other items used as missiles in the protest context.

- **memorial_elements**: Commemorative items placed to honor or remember someone, including candles, flowers, photographs, crosses, wreaths, or other tribute objects.

- **location_unclear**: The setting or location of the image cannot be clearly determined due to limited visual information, poor image quality, obstruction, or ambiguous environmental cues.

- **mutual_cover**: Mutual Cover or Shielding is an act of physical protection in which one person intentionally shields, blocks, or defends another individual from perceived threat or danger. This may be done with a body, object (e.g., shield, umbrella, backpack).

- **standing_in_line_or_wall_formation**: Standing in line or wall formation is a formation in which people stand in a line or wall formation, typically with their hands on the shoulders of the person in front of or next to them.

- **police_solidarity_degree**: Rate the level of visible solidarity among police officers in this image from 0 to 10. If your score is between 6 and 10, return "high". If between 1 and 5, return "low". If 0, return "none". Return only: police_solidarity_degree: [high/low/none]

- **protester_solidarity_degree**: Rate the level of visible solidarity among protesters in this image from 0 to 10. If your score is between 6 and 10, return "high". If between 1 and 5, return "low". If 0, return "none". Return only: protester_solidarity_degree: [high/low/none]

- **conflict_degree**: Rate the level of visible conflict between protesters and police officers, or between protesters and other protesters, or between protesters and the public in this image from 0 to 10. If your score is between 6 and 10, return "high". If between 1 and 5, return "low". If 0, return "none". Return only: conflict_degree: [high/low/none]

"""

prompt = """Analyze this image of a protest and determine:

1. Identify elements in the specific categories below.
2. Does it show CONFLICT as defined? If true, list the elements from step 1 supporting the conflict framing; if false, leave the list, "conflict_supporting_elements", empty. 
3. Does it show PEACE as defined? If true, list the elements from step 1 supporting the peace framing; if false, leave the list, "peace_supporting_elements", empty.
4. Does it show PROTESTER SOLIDARITY as defined? If true, list the elements from step 1 supporting the protester solidarity framing; if false, leave the list, "protester_solidarity_supporting_elements", empty.
5. Does it show POLICE SOLIDARITY as defined? If true, list the elements from step 1 supporting the police solidarity framing; if false, leave the list, "police_solidarity_supporting_elements", empty.

If any category cannot be determined from the image, use false for booleans. DO NOT GUESS.

Respond in JSON format ONLY:

{
"people": {
    "number_of_people": [number],
    "crowd_size_none": true/false,
    "crowd_size_small_1_to_10": true/false,
    "crowd_size_medium_11_to_50": true/false,
    "crowd_size_large_50_plus": true/false,
    "crowd_dynamics": {
        "crowd_dynamics_chaotic_or_disorganized": true/false,
        "crowd_dynamics_organized_or_unified": true/false
    },
    "police_present": true/false,
    "police_regular_gear": true/false,
    "police_riot_gear": true/false,
    "children_present": true/false
},
"actions": {
    "marching": true/false,
    "standing_confrontation": true/false,
    "action_fighting": true/false,
    "action_throwing_objects": true/false,
    "action_shouting": true/false,
    "damaging_property": true/false,
    "arrests_visible": true/false,
    "protester_violence": true/false,
    "police_using_force": true/false,
    "medical_aid": true/false,
    "speaking_or_chanting": true/false,
    "aggressive_gestures": true/false,
    "raised_fists_or_hands": true/false,
    "pointing_fingers": true/false,
    "kneeling": true/false,
    "linked_arms": true/false,
    "holding_hands": true/false,
    "peaceful_gathering": true/false,
    "vigils": true/false,
    "comforting_or_hugging": true/false,
    "retreating_or_running_from_police": true/false,
    "shields_raised": true/false,
    "firearms_raised": true/false,
    "distributing_supplies": true/false,
    "helping_injured": true/false,
    "batons_raised": true/false,
    "mutual_cover": true/false,
    "standing_in_line_or_wall_formation": true/false
},
"protester_emotions": {
    "happy": true/false,
    "angry": true/false,
    "somber": true/false,
    "determined": true/false,
    "fearful": true/false,
    "calm": true/false,
    "tense": true/false
},
"police_emotions": {
    "happy": true/false,
    "angry": true/false,
    "somber": true/false,
    "determined": true/false,
    "fearful": true/false,
    "calm": true/false,
    "tense": true/false
},
"objects": {
    "weapons_visible": true/false,
    "projectiles": true/false,
    "barriers_or_fences": true/false,
    "shields": true/false,
    "batons": true/false,
    "signs_or_banners": true/false,
    "flags": true/false,
    "burning_or_trampled_flag": true/false,
    "megaphones_or_speakers": true/false,
    "cameras_or_phones": true/false,
    "injured_or_dead_bodies": true/false,
    "vehicle_none": true/false,
    "vehicle_civilian": true/false,
    "vehicle_police": true/false,
    "vehicle_emergency": true/false,
    "vehicle_unclear": true/false,
    "damage_visible": true/false,
    "graffiti": true/false,
    "smoke_or_fire": true/false,
    "debris_trash_or_garbage": true/false,
    "debris_glass_or_broken_glass": true/false,
    "memorial_elements": true/false,
    "umbrellas_or_improvised_shields": true/false
},
"environment": {
    "location_outdoor": true/false,
    "location_indoor": true/false,
    "location_unclear": true/false
},
"police_solidarity": true/false,
"police_solidarity_supporting_elements": ["list", "of", "elements", "supporting", "police", "solidarity", "from", "the", "objects", "category", "above"],
"police_solidarity_degree": none/low/high,
"protester_solidarity": true/false,
"protester_solidarity_supporting_elements": ["list", "of", "elements", "supporting", "protester", "solidarity", "from", "the", "objects", "category", "above"],
"protester_solidarity_degree": none/low/high,
"peace": true/false,
"peace_supporting_elements": ["list", "of", "elements", "supporting", "peace", "from", "the", "objects", "category", "above"],
"conflict": true/false,
"conflict_supporting_elements": ["list", "of", "elements", "supporting", "conflict", "from", "the", "objects", "category", "above"],
"conflict_degree": none/low/high,
}"""

def get_model_config(model_name: str) -> Dict[str, any]:
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, {
        "temperature": 0.1,
        "num_ctx": 8192,
        "num_predict": 2000
    })

def get_api_config(api_type: str, model_name: str) -> Dict[str, any]:
    """Get configuration for a specific model and API"""
    return API_CONFIGS.get(api_type, {}).get(model_name, {
        "temperature": 0.1,
        "max_tokens": 2000
    })

def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from response text, handling markdown code blocks"""
    # Remove markdown code blocks if present
    if "```json" in response_text:
        # Extract content between ```json and ```
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        if end != -1:
            return response_text[start:end].strip()
    elif "```" in response_text:
        # Handle generic code blocks
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        if end != -1:
            return response_text[start:end].strip()
    
    # If no code blocks, try to find JSON-like content
    # Look for content between { and }
    start = response_text.find("{")
    if start != -1:
        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(response_text[start:], start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return response_text[start:i+1]
    
    # Return original if no JSON structure found
    return response_text.strip()

def analyze_image_with_vllm(image_path: str, model: str = "OpenGVLab/InternVL3-14B", 
                           vllm_url: str = "http://localhost:8001") -> Dict[str, any]:
    """Analyze image using vLLM API (OpenAI-compatible)"""

    try:
        # Initialize OpenAI client pointing to vLLM server
        client = openai.OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=f"{vllm_url}/v1"
        )
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Get model-specific configuration
        model_config = get_api_config("vllm", model)
        
        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": previous_background_knowledge
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Make API call with longer timeout for vLLM
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **model_config
        )
        
        # Extract and parse response
        response_text = response.choices[0].message.content
        try:
            response_text = extract_json_from_response(response_text)
            analysis = json.loads(response_text)
            analysis['raw_llm_output'] = response_text  # Store raw output
            return analysis
        except json.JSONDecodeError:
            print(f"\n\nFailed to parse JSON for {image_path}: {response_text}\n\n")
            empty_analysis = create_empty_analysis()
            empty_analysis['raw_llm_output'] = response_text  # Store raw output even if parsing fails
            return empty_analysis
            
    except Exception as e:
        print(f"Error analyzing {image_path} with vLLM: {e}")
        return create_empty_analysis()

def analyze_image_with_ollama(image_path: str, model: str = "gemma3:27b-it-q8_0") -> Dict[str, any]:
    """Analyze image using Ollama API"""
    # Encode the image
    base64_image = encode_image(image_path)
    
    # Get model-specific configuration
    model_config = get_api_config("ollama", model)

    # Prepare the request to Ollama
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": previous_background_knowledge + "\n\n" + prompt,
        "images": [base64_image],
        "stream": False,
        "format": "json",
        "keep_alive": "1m",
        "options": model_config
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Extract the response
        result = response.json()
        response_text = result.get('response', '{}')
        
        # Parse JSON response
        try:
            response_text = extract_json_from_response(response_text)
            analysis = json.loads(response_text)
            analysis['raw_llm_output'] = response_text  # Store raw output
            return analysis
        except json.JSONDecodeError:
            print(f"\n\nFailed to parse JSON for {image_path}: {response_text}\n\n")
            empty_analysis = create_empty_analysis()
            empty_analysis['raw_llm_output'] = response_text  # Store raw output even if parsing fails
            return empty_analysis
            
    except requests.exceptions.RequestException as e:
        print(f"Error analyzing {image_path}: {e}")
        return create_empty_analysis()

def analyze_image_with_openai(image_path: str, model: str = "gpt-4-vision-preview") -> Dict[str, any]:
    """Analyze image using OpenAI API"""
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Get model-specific configuration
        model_config = get_api_config("openai", model)
        
        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": previous_background_knowledge
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **model_config
        )
        
        # Extract and parse response
        response_text = response.choices[0].message.content
        try:
            response_text = extract_json_from_response(response_text)
            analysis = json.loads(response_text)
            analysis['raw_llm_output'] = response_text  # Store raw output
            return analysis
        except json.JSONDecodeError:
            print(f"\n\nFailed to parse JSON for {image_path}: {response_text}\n\n")
            empty_analysis = create_empty_analysis()
            empty_analysis['raw_llm_output'] = response_text  # Store raw output even if parsing fails
            return empty_analysis
            
    except Exception as e:
        print(f"Error analyzing {image_path} with OpenAI: {e}")
        return create_empty_analysis()

def analyze_image(image_path: str, api_type: str = "ollama", model: str = "gemma3:27b-it-q8_0",
                  vllm_url: str = "http://localhost:8001") -> Dict[str, any]:
    """Analyze image using specified API and model"""
    if api_type == "ollama":
        return analyze_image_with_ollama(image_path, model)
    elif api_type == "openai":
        return analyze_image_with_openai(image_path, model)
    elif api_type == "vllm":
        return analyze_image_with_vllm(image_path, model, vllm_url)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")

def create_empty_analysis() -> Dict[str, any]:
    """Create an empty analysis structure with default values"""
    return {
        "raw_llm_output": "",  # Add raw LLM output field
        "people": {
            "number_of_people": 0,
            "crowd_size_none": False,
            "crowd_size_small_1_to_10": False,
            "crowd_size_medium_11_to_50": False,
            "crowd_size_large_50_plus": False,
            "crowd_dynamics": {
                "crowd_dynamics_chaotic_or_disorganized": False,
                "crowd_dynamics_organized_or_unified": False
            },
            "police_present": False,
            "police_regular_gear": False,
            "police_riot_gear": False,
            "children_present": False,
        },
        "actions": {
            "marching": False,
            "standing_confrontation": False,
            "action_fighting": False,
            "action_throwing_objects": False,
            "action_shouting": False,
            "damaging_property": False,
            "arrests_visible": False,
            "protester_violence": False,
            "police_using_force": False,
            "medical_aid": False,
            "speaking_or_chanting": False,
            "aggressive_gestures": False,
            "raised_fists_or_hands": False,
            "pointing_fingers": False,
            "kneeling": False,
            "linked_arms": False,
            "holding_hands": False,
            "peaceful_gathering": False,
            "vigils": False,
            "comforting_or_hugging": False,
            "retreating_or_running_from_police": False,
            "shields_raised": False,
            "firearms_raised": False,
            "distributing_supplies": False,
            "helping_injured": False,
            "batons_raised": False,
            "mutual_cover": False,
            "standing_in_line_or_wall_formation": False
        },
        "protester_emotions": {
            "happy": False,
            "angry": False,
            "somber": False,
            "determined": False,
            "fearful": False,
            "calm": False,
            "tense": False
        },
        "police_emotions": {
            "happy": False,
            "angry": False,
            "somber": False,
            "determined": False,
            "fearful": False,
            "calm": False,
            "tense": False
        },
        "objects": {
            "weapons_visible": False,
            "projectiles": False,
            "barriers_or_fences": False,
            "shields": False,
            "batons": False,
            "signs_or_banners": False,
            "flags": False,
            "burning_or_trampled_flag": False,
            "megaphones_or_speakers": False,
            "cameras_or_phones": False,
            "injured_or_dead_bodies": False,
            "vehicle_none": False,
            "vehicle_civilian": False,
            "vehicle_police": False,
            "vehicle_emergency": False,
            "vehicle_unclear": False,
            "damage_visible": False,
            "graffiti": False,
            "smoke_or_fire": False,
            "debris_trash_or_garbage": False,
            "debris_glass_or_broken_glass": False,
            "memorial_elements": False,
            "umbrellas_or_improvised_shields": False
        },
        "environment": {
            "location_outdoor": False,
            "location_indoor": False,
            "location_unclear": False
        },
        "police_solidarity": False,
        "police_solidarity_supporting_elements": [],
        "police_solidarity_degree": "none",
        "protester_solidarity": False,
        "protester_solidarity_supporting_elements": [],
        "protester_solidarity_degree": "none",
        "peace": False,
        "peace_supporting_elements": [],
        "conflict": False,
        "conflict_supporting_elements": [],
        "conflict_degree": "none"
    }

def flatten_analysis(analysis: Dict[str, any], filename: str) -> Dict[str, any]:
    """Flatten the nested analysis structure for CSV writing"""
    flat_dict = {}
    
    # Handle basic fields with error handling
    try:
        flat_dict['filename'] = filename
    except Exception as e:
        print(f"Error processing filename for {filename}: {e}")
        flat_dict['filename'] = filename  # Keep original filename even if there's an error
    
    # Handle boolean fields
    boolean_fields = ['police_solidarity', 'protester_solidarity', 'peace', 'conflict']
    for field in boolean_fields:
        try:
            flat_dict[field] = analysis.get(field, False)
        except Exception as e:
            print(f"Error processing {field} for {filename}: {e}")
            flat_dict[field] = False
    
    # Handle degree fields
    degree_fields = ['police_solidarity_degree', 'protester_solidarity_degree', 'conflict_degree']
    for field in degree_fields:
        try:
            flat_dict[field] = analysis.get(field, 'none')
        except Exception as e:
            print(f"Error processing {field} for {filename}: {e}")
            flat_dict[field] = 'none'
    
    # Handle supporting elements fields
    supporting_fields = [
        'police_solidarity_supporting_elements',
        'protester_solidarity_supporting_elements',
        'peace_supporting_elements',
        'conflict_supporting_elements'
    ]
    for field in supporting_fields:
        try:
            elements = analysis.get(field, [])
            flat_dict[field] = '; '.join(elements) if isinstance(elements, list) else str(elements)
        except Exception as e:
            print(f"Error processing {field} for {filename}: {e}")
            flat_dict[field] = ''
    
    # Handle raw LLM output
    try:
        flat_dict['raw_llm_output'] = analysis.get('raw_llm_output', '')
    except Exception as e:
        print(f"Error processing raw_llm_output for {filename}: {e}")
        flat_dict['raw_llm_output'] = ''
    
    # Flatten people category
    people_data = analysis.get('people', {})
    for key, value in people_data.items():
        try:
            if key == 'crowd_dynamics':
                crowd_dynamics = value if isinstance(value, dict) else {}
                for subkey, subvalue in crowd_dynamics.items():
                    try:
                        flat_dict[f'people_{subkey}'] = subvalue
                    except Exception as e:
                        print(f"Error processing people_crowd_dynamics_{subkey} for {filename}: {e}")
                        flat_dict[f'people_{subkey}'] = False
            else:
                flat_dict[f'people_{key}'] = value
        except Exception as e:
            print(f"Error processing people_{key} for {filename}: {e}")
            flat_dict[f'people_{key}'] = False
    
    # Flatten other categories
    categories = ['actions', 'protester_emotions', 'police_emotions', 'objects', 'environment']
    for category in categories:
        category_data = analysis.get(category, {})
        for key, value in category_data.items():
            try:
                flat_dict[f'{category}_{key}'] = value
            except Exception as e:
                print(f"Error processing {category}_{key} for {filename}: {e}")
                flat_dict[f'{category}_{key}'] = False
    
    return flat_dict

def process_images(image_folder: str, output_csv: str = "protest_analysis.csv", 
                  api_type: str = "ollama", model: str = "gemma3:27b-it-q8_0",
                  vllm_url: str = "http://localhost:8001", sample_size: int = 200):
    """
    Process all images in a folder and save results to CSV
    """
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = []
    
    folder_path = Path(image_folder)
    for file in folder_path.iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    # randomly choose sample_size images
    image_files = random.sample(image_files, min(sample_size, len(image_files)))

    print(f"Found {len(image_files)} images to process using {api_type} API with model: {model}")
    
    # Get fieldnames from a sample analysis
    sample_analysis = create_empty_analysis()
    fieldnames = list(flatten_analysis(sample_analysis, '').keys())
    
    # Prepare CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each image with or without progress bar based on debug flag
        iterator = tqdm(image_files, desc="Processing images", unit="image") if USE_TQDM else image_files
        for i, image_path in enumerate(iterator):
            # Analyze image
            result = analyze_image(str(image_path), api_type, model, vllm_url)
            
            # Flatten the result and add filename
            flat_result = flatten_analysis(result, str(image_path.name))
            
            # Write to CSV
            writer.writerow(flat_result)
            
            # Save CSV every 10 images
            save_every = 10 if USE_TQDM else 1
            if (i + 1) % save_every == 0:
                csvfile.flush()
                if USE_TQDM:
                    tqdm.write(f"Saved progress after {i+1} images")
                else:
                    print(f"Saved progress after {i+1} images")
            
            # Rate limiting to avoid overwhelming APIs - increased for vLLM
            if api_type == "vllm":
                time.sleep(3)  # Longer delay for vLLM to prevent timeouts
            else:
                time.sleep(1)
    
    print(f"Analysis complete! Results saved to {output_csv}")

def check_vllm_server(vllm_url: str = "http://localhost:8001") -> bool:
    """Check if vLLM server is running and accessible"""
    try:
        response = requests.get(f"{vllm_url}/health")
        return response.status_code == 200
    except:
        try:
            # Alternative check - try to get models
            response = requests.get(f"{vllm_url}/v1/models")
            return response.status_code == 200
        except:
            return False

def main():
    """
    Main function to run the analysis
    """
    # Parse command line arguments
    args = parse_args()
    
    # Update global debug flag if progress bar is disabled
    global USE_TQDM
    if args.no_progress_bar:
        USE_TQDM = False
    
    # Check API availability
    if args.api == "ollama":
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            models = [m['name'] for m in response.json().get('models', [])]
            
            if args.model not in models:
                print(f"Warning: Model {args.model} not found. Available models: {models}")
                print(f"Please pull the model first: ollama pull {args.model}")
                return
        except:
            print("Error: Cannot connect to Ollama. Make sure it's running (ollama serve)")
            return
    elif args.api == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found in environment variables")
            return
    elif args.api == "vllm":
        if not check_vllm_server(args.vllm_url):
            print(f"Error: Cannot connect to vLLM server at {args.vllm_url}")
            print("Make sure vLLM is running with:")
            print(f"TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 vllm serve {args.model} --port 8001")
            return
        else:
            print(f"✓ Connected to vLLM server at {args.vllm_url}")
            time.sleep(3)
    
    # Process images
    process_images(
        image_folder=args.image_folder,
        output_csv=args.output_csv,
        api_type=args.api,
        model=args.model,
        vllm_url=args.vllm_url,
        sample_size=args.sample_size
    )

if __name__ == "__main__":
    main()