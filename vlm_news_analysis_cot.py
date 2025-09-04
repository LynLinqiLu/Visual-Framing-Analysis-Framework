"""
Image Analysis Script for Protest Images

This script analyzes protest images using various AI models (Ollama, OpenAI, vLLM)
with support for Chain-of-Thought reasoning. It can classify images based on
conflict, peace, protester solidarity, and police solidarity frames.

Features:
- Multiple API support (Ollama, OpenAI, vLLM)
- Chain-of-Thought reasoning with configurable depth
- Confidence scoring and calibration
- Batch processing with progress tracking
- Resume capability for interrupted runs

Usage:
    python vlm_news_analysis_cot_resume_feature.py --image-folder ./images --api ollama
"""

import os
import csv
import base64
import requests
import json
from pathlib import Path
from typing import Dict
import time
import random
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Global flag for progress bar display
USE_TQDM = True

def parse_args():
    """Parse command line arguments for the image analysis script."""
    parser = argparse.ArgumentParser(description='Analyze protest images using various AI models')
    
    # API and model selection
    parser.add_argument('--api', type=str, default='ollama',
                      choices=['ollama', 'openai', 'vllm'],
                      help='API type to use (default: ollama)')
    parser.add_argument('--model', type=str, default='gemma3:27b-it-q8_0',
                      help='Model to use (default: gemma3:27b-it-q8_0)')
    
    # Chain-of-Thought options
    parser.add_argument('--use-cot', action='store_true',
                      help='Enable Chain-of-Thought reasoning')
    parser.add_argument('--cot-depth', type=str, default='detailed',
                      choices=['simple', 'detailed', 'expert'],
                      help='CoT reasoning depth (default: detailed)')
    
    # File paths
    parser.add_argument('--image-folder', type=str,
                      default='./images',
                      help='Path to folder containing images to analyze')
    parser.add_argument('--output-csv', type=str,
                      default='protest_analysis_enhanced.csv',
                      help='Path to output CSV file')
    
    # vLLM specific
    parser.add_argument('--vllm-url', type=str, default='http://localhost:8001',
                      help='URL of vLLM server (default: http://localhost:8001)')
    parser.add_argument('--timeout', type=int, default=600,
                      help='Timeout in seconds for API calls (default: 600)')
    
    # Processing options
    parser.add_argument('--no-progress-bar', action='store_true',
                      help='Disable progress bar')
    parser.add_argument('--sample-size', type=int, default=200,
                      help='Number of images to randomly sample (default: 200)')
    
    return parser.parse_args()

# Model configurations with confidence calibration
MODEL_CONFIGS = {
    "gemma3:27b-it-q8_0": {
        "temperature": 0.1,
        "num_ctx": 12288,
        "num_predict": 3000,
        "api_type": "ollama"
    },
    "qwen2.5vl:72b-q4_K_M": {
        "temperature": 0.1,
        "num_ctx": 12288,
        "num_predict": 3000,
        "api_type": "ollama"
    },
    "OpenGVLab/InternVL3-14B": {
        "temperature": 0.1,
        "max_tokens": 3000,
        "api_type": "vllm"
    },
    "OpenGVLab/InternVL3-38B": {
        "temperature": 0.1,
        "max_tokens": 3000,
        "api_type": "vllm"
    },
    "gpt-4-vision-preview": {
        "temperature": 0.1,
        "max_tokens": 3000,
        "api_type": "openai"
    }
}

# Chain-of-Thought prompts for different reasoning depths
COT_PROMPTS = {
    "simple": """
**Step 1: Initial Observation**
Let me examine this protest image systematically:
- Crowd characteristics: [describe what you observe]
- Key visual elements: [list specific items you notice]

**Step 2: Frame Analysis**
Based on my observations:
- CONFLICT indicators: [specific evidence]
- PEACE indicators: [specific evidence]
- PROTESTER SOLIDARITY evidence: [specific behaviors]
- POLICE SOLIDARITY evidence: [specific officer interactions]

**Step 3: Final Classification**
Now I'll encode these observations in the required JSON format:
""",
    
    "detailed": """
**Step 1: Visual Survey**
Let me systematically examine this protest image:
- Crowd characteristics: [describe what you observe]
- Key visual elements: [list specific items you notice]
- Spatial relationships: [describe positioning and interactions]

**Step 2: Behavioral Analysis**  
Now analyzing human dynamics:
- Emotional indicators: [specific facial expressions, body language]
- Interaction patterns: [how groups relate to each other]
- Evidence of tension or cooperation: [specific behaviors observed]

**Step 3: Symbolic Interpretation**
Examining signs, symbols, and contextual elements:
- Textual messages: [what signs/banners say and imply]
- Visual symbols: [flags, gestures, clothing and their meanings]
- Environmental context: [setting and its significance]

**Step 4: Frame Classification**
Based on my systematic analysis:
- CONFLICT indicators: [specific evidence from steps 1-3]
- PEACE indicators: [specific evidence from steps 1-3]  
- PROTESTER SOLIDARITY evidence: [specific behaviors and symbols]
- POLICE SOLIDARITY evidence: [specific officer interactions]

**Step 5: Confidence Assessment**
Rate my confidence in each classification (0-10):
- Conflict frame confidence: [score with reasoning]
- Peace frame confidence: [score with reasoning]
- Protester solidarity confidence: [score with reasoning]
- Police solidarity confidence: [score with reasoning]

**Step 6: Final JSON Output**
Now I'll encode these observations in the required format:
""",
    
    "expert": """
**Step 1: Professional Visual Assessment**
As an expert in social movement analysis, let me examine this protest image:
- Crowd dynamics and density patterns: [detailed analysis]
- Spatial organization and territorial claims: [specific observations]
- Visual hierarchy and focus points: [key elements drawing attention]

**Step 2: Sociological Context Analysis**
Analyzing the social dynamics:
- Power relationships visible in positioning: [specific evidence]
- Group cohesion indicators: [behavioral observations]
- Authority-civilian interaction patterns: [detailed description]
- Emotional climate assessment: [facial expressions, body language, energy]

**Step 3: Symbolic and Semiotic Analysis**
Examining communicative elements:
- Textual messaging strategies: [signs, banners, their positioning and design]
- Visual symbolism deployment: [flags, colors, gestures, their cultural meanings]
- Performative elements: [ritualistic behaviors, collective actions]
- Counter-messaging or competing narratives: [conflicting symbols or messages]

**Step 4: Frame Theory Application**
Applying established framing theory to classify this image:
- CONFLICT FRAME evidence: [specific visual elements that create confrontational narrative]
  * Escalation indicators: [behaviors suggesting rising tension]
  * Oppositional positioning: [spatial arrangements suggesting conflict]
  * Aggressive symbolism: [objects, gestures, expressions indicating hostility]
  
- PEACE FRAME evidence: [specific visual elements that create peaceful narrative]  
  * De-escalation behaviors: [calming gestures, peaceful postures]
  * Collaborative positioning: [spatial arrangements suggesting cooperation]
  * Unity symbolism: [shared symbols, harmonious arrangements]
  
- PROTESTER SOLIDARITY evidence: [collective action indicators]
  * Mutual aid behaviors: [specific supportive actions between protesters]
  * Synchronized actions: [coordinated movements, shared behaviors]
  * Protective formations: [evidence of protesters protecting each other]
  
- POLICE SOLIDARITY evidence: [law enforcement unity indicators]
  * Inter-officer support: [specific cooperative behaviors between officers]
  * Coordinated responses: [synchronized police actions]
  * Professional camaraderie: [evidence of mutual support among officers]

**Step 5: Methodological Confidence Assessment**
Evaluating the strength of evidence for each frame (scale 0-10):
- Visual clarity and unambiguous evidence: [assessment of image quality and clarity]
- Consistency across multiple indicators: [how well different elements support same conclusion]
- Absence of contradictory evidence: [checking for elements that undermine classification]
- Confidence scores with detailed justification:
  * Conflict: [score] because [specific reasoning]
  * Peace: [score] because [specific reasoning]  
  * Protester solidarity: [score] because [specific reasoning]
  * Police solidarity: [score] because [specific reasoning]

**Step 6: Final Structured Output**
Based on this comprehensive analysis, I'll now provide the structured JSON classification:
"""
}

# Background knowledge for image analysis
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

def create_enhanced_prompt(use_cot: bool = False, cot_depth: str = "detailed") -> str:
    """Create analysis prompt with optional Chain-of-Thought reasoning."""
    base_prompt = """Analyze this image of a protest and determine:

1. Identify elements in the specific categories below.
2. Does it show CONFLICT as defined? If true, list the elements from step 1 supporting the conflict framing; if false, leave the list, "conflict_supporting_elements", empty. 
3. Does it show PEACE as defined? If true, list the elements from step 1 supporting the peace framing; if false, leave the list, "peace_supporting_elements", empty.
4. Does it show PROTESTER SOLIDARITY as defined? If true, list the elements from step 1 supporting the protester solidarity framing; if false, leave the list, "protester_solidarity_supporting_elements", empty.
5. Does it show POLICE SOLIDARITY as defined? If true, list the elements from step 1 supporting the police solidarity framing; if false, leave the list, "police_solidarity_supporting_elements", empty.

If any category cannot be determined from the image, use false for booleans. DO NOT GUESS."""

    if use_cot:
        cot_section = COT_PROMPTS.get(cot_depth, COT_PROMPTS["detailed"])
        return base_prompt + "\n\n" + cot_section + "\n\nRespond in JSON format ONLY:"
    else:
        return base_prompt + "\n\nRespond in JSON format ONLY:"

# JSON schema for analysis output
def get_json_schema() -> str:
    return """
{
"reasoning_chain": "If CoT enabled, include your step-by-step reasoning here",
"confidence_scores": {
    "conflict": 0-10,
    "peace": 0-10,
    "protester_solidarity": 0-10,
    "police_solidarity": 0-10
},
"people": {
    "number_of_people": "0",
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
"police_solidarity_supporting_elements": ["list", "of", "elements"],
"police_solidarity_degree": none/low/high,
"protester_solidarity": true/false,
"protester_solidarity_supporting_elements": ["list", "of", "elements"],
"protester_solidarity_degree": none/low/high,
"peace": true/false,
"peace_supporting_elements": ["list", "of", "elements"],
"conflict": true/false,
"conflict_supporting_elements": ["list", "of", "elements"],
"conflict_degree": none/low/high
}"""

def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from LLM response with error handling."""
    try:
        # Remove markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                json_str = response_text[start:end].strip()
                json.loads(json_str)  # Validate JSON
                return json_str
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                json_str = response_text[start:end].strip()
                json.loads(json_str)
                return json_str
        
        # Look for JSON content between { and }
        start = response_text.find("{")
        if start != -1:
            brace_count = 0
            for i, char in enumerate(response_text[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response_text[start:i+1]
                        json.loads(json_str)
                        return json_str
        
        return "{}"
        
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in response: {e}")
        return "{}"
    except Exception as e:
        print(f"Warning: Error extracting JSON: {e}")
        return "{}"

def extract_confidence_scores(analysis: Dict) -> Dict[str, float]:
    """Extract confidence scores from analysis with fallbacks."""
    confidence_scores = analysis.get('confidence_scores', {})
    
    default_scores = {
        'conflict': 5.0,
        'peace': 5.0,
        'protester_solidarity': 5.0,
        'police_solidarity': 5.0
    }
    
    for key in default_scores:
        if key not in confidence_scores:
            confidence_scores[key] = default_scores[key]
        else:
            confidence_scores[key] = max(0, min(10, float(confidence_scores[key])))
    
    return confidence_scores

def analyze_image_enhanced(image_path: str, model: str, use_cot: bool = False, 
                         cot_depth: str = "detailed", api_type: str = None,
                         vllm_url: str = "http://localhost:8001", timeout: int = 600) -> Dict[str, any]:
    """Analyze image using specified model with optional Chain-of-Thought reasoning."""
    
    model_config = MODEL_CONFIGS.get(model, {})
    if api_type is None:
        api_type = model_config.get('api_type', 'ollama')
    
    prompt_text = create_enhanced_prompt(use_cot, cot_depth)
    full_prompt = prompt_text + "\n\n" + get_json_schema()
    
    try:
        if api_type == "ollama":
            return analyze_image_with_ollama_enhanced(image_path, model, full_prompt)
        elif api_type == "openai":
            return analyze_image_with_openai_enhanced(image_path, model, full_prompt)
        elif api_type == "vllm":
            return analyze_image_with_vllm_enhanced(image_path, model, full_prompt, vllm_url, timeout)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
            
    except Exception as e:
        print(f"Error in enhanced analysis for {image_path}: {e}")
        return create_empty_analysis()

def analyze_image_with_ollama_enhanced(image_path: str, model: str, prompt: str) -> Dict[str, any]:
    """Analyze image using Ollama API."""
    base64_image = encode_image(image_path)
    model_config = MODEL_CONFIGS.get(model, {})
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": previous_background_knowledge + "\n\n" + prompt,
        "images": [base64_image],
        "stream": False,
        "format": "json",
        "keep_alive": "5m",
        "options": {k: v for k, v in model_config.items() if k != 'api_type'}
    }
    
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '{}')
        
        response_text = extract_json_from_response(response_text)
        analysis = json.loads(response_text)
        analysis['raw_llm_output'] = response_text
        analysis['model_used'] = model
        
        return analysis
        
    except Exception as e:
        print(f"Error in Ollama analysis for {image_path}: {e}")
        return create_empty_analysis()

def analyze_image_with_openai_enhanced(image_path: str, model: str, prompt: str) -> Dict[str, any]:
    """Analyze image using OpenAI API."""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        model_config = MODEL_CONFIGS.get(model, {})
        
        messages = [
            {"role": "system", "content": previous_background_knowledge},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=model_config.get('temperature', 0.1),
            max_tokens=model_config.get('max_tokens', 3000)
        )
        
        response_text = response.choices[0].message.content
        response_text = extract_json_from_response(response_text)
        analysis = json.loads(response_text)
        analysis['raw_llm_output'] = response_text
        analysis['model_used'] = model
        
        return analysis
        
    except Exception as e:
        print(f"Error in OpenAI analysis for {image_path}: {e}")
        return create_empty_analysis()

def analyze_image_with_vllm_enhanced(image_path: str, model: str, prompt: str, vllm_url: str, timeout: int = 600) -> Dict[str, any]:
    """Analyze image using vLLM API with timeout handling."""
    try:
        client = openai.OpenAI(api_key="EMPTY", base_url=f"{vllm_url}/v1", timeout=timeout)
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        model_config = MODEL_CONFIGS.get(model, {})
        
        messages = [
            {"role": "system", "content": previous_background_knowledge},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=model_config.get('temperature', 0.1),
            max_tokens=model_config.get('max_tokens', 3000)
        )
        
        response_text = response.choices[0].message.content
        response_text = extract_json_from_response(response_text)
        analysis = json.loads(response_text)
        analysis['raw_llm_output'] = response_text
        analysis['model_used'] = model
        
        return analysis
        
    except Exception as e:
        print(f"Error in vLLM analysis for {image_path}: {e}")
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['timeout', 'timed out', 'connection timeout', 'read timeout']):
            print(f"Timeout detected for {image_path}")
            return None
        else:
            return None









def encode_image(image_path: str) -> str:
    """Encode image to base64 string with error handling."""
    try:
        with open(image_path, "rb") as image_file:
            try:
                return base64.b64encode(image_file.read()).decode('utf-8')
            except UnicodeDecodeError:
                print(f"Warning: Failed to decode image {image_path}")
                return ""
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return ""
    except PermissionError:
        print(f"Error: Permission denied reading image: {image_path}")
        return ""
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return ""

def create_empty_analysis() -> Dict[str, any]:
    """Create an empty analysis structure with default values."""
    return {
        "raw_llm_output": "",
        "model_used": "unknown",
        "reasoning_chain": "",
        "confidence_scores": {
            "conflict": 0,
            "peace": 0,
            "protester_solidarity": 0,
            "police_solidarity": 0
        },
        "people": {
            "number_of_people": "0",
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

def flatten_analysis_enhanced(analysis: Dict[str, any], filename: str) -> Dict[str, any]:
    """Flatten nested analysis structure for CSV output with type safety."""
    flat_dict = {}
    
    def safe_get_dict(data: any, default: dict = None) -> dict:
        if isinstance(data, dict):
            return data
        return default or {}
    
    def safe_get_list(data: any, default: list = None) -> list:
        if isinstance(data, list):
            return data
        return default or []
    
    try:
        flat_dict['filename'] = filename
    except Exception as e:
        print(f"Error processing filename for {filename}: {e}")
        flat_dict['filename'] = filename
    
    flat_dict['model_used'] = analysis.get('model_used', 'unknown')
    
    confidence_scores = safe_get_dict(analysis.get('confidence_scores'))
    for frame in ['conflict', 'peace', 'protester_solidarity', 'police_solidarity']:
        try:
            flat_dict[f'confidence_{frame}'] = float(confidence_scores.get(frame, 0))
        except (ValueError, TypeError):
            flat_dict[f'confidence_{frame}'] = 0.0
    
    flat_dict['reasoning_chain'] = analysis.get('reasoning_chain', '')[:1000]
    
    boolean_fields = ['police_solidarity', 'protester_solidarity', 'peace', 'conflict']
    for field in boolean_fields:
        try:
            flat_dict[field] = analysis.get(field, False)
        except Exception as e:
            print(f"Error processing {field} for {filename}: {e}")
            flat_dict[field] = False
    
    degree_fields = ['police_solidarity_degree', 'protester_solidarity_degree', 'conflict_degree']
    for field in degree_fields:
        try:
            flat_dict[field] = analysis.get(field, 'none')
        except Exception as e:
            print(f"Error processing {field} for {filename}: {e}")
            flat_dict[field] = 'none'
    
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
    
    try:
        flat_dict['raw_llm_output'] = analysis.get('raw_llm_output', '')
    except Exception as e:
        print(f"Error processing raw_llm_output for {filename}: {e}")
        flat_dict['raw_llm_output'] = ''
    
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

def process_images_enhanced(image_folder: str, output_csv: str = "protest_analysis_enhanced.csv", 
                          api_type: str = "ollama", model: str = "gemma3:27b-it-q8_0",
                          use_cot: bool = False, cot_depth: str = "detailed",
                          vllm_url: str = "http://localhost:8001", 
                          sample_size: int = 200, timeout: int = 600):
    """Process images with Chain-of-Thought reasoning support."""
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = []
    
    folder_path = Path(image_folder)
    for file in folder_path.iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    print(f"Found {len(image_files)} images to process")
    
    timeout_files = []
    if os.path.exists('timeout_files.txt'):
        try:
            with open('timeout_files.txt', 'r') as f:
                timeout_files = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(timeout_files)} timeout files from timeout_files.txt")
        except Exception as e:
            print(f"Warning: Could not read timeout_files.txt: {e}")
    
    print(f"Using model: {model}")
    
    if use_cot:
        print(f"Chain-of-Thought enabled with depth: {cot_depth}")
    
    sample_analysis = create_empty_analysis()
    fieldnames = list(flatten_analysis_enhanced(sample_analysis, '').keys())
    
    processed_files = set()
    mode = 'w'
    
    if os.path.exists(output_csv):
        try:
            with open(output_csv, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if 'filename' in row:
                        processed_files.add(row['filename'])
            if processed_files:
                print(f"Found {len(processed_files)} previously processed images")
                mode = 'a'
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {e}")
            print("Starting fresh analysis")
            mode = 'w'
    
    remaining_images = [img for img in image_files if img.name not in processed_files and img.name not in timeout_files]
    print(f"Remaining images to process: {len(remaining_images)}")
    
    if not remaining_images:
        print("All images have been processed!")
        return
    
    with open(output_csv, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        
        iterator = tqdm(remaining_images, desc="Processing images", unit="image") if USE_TQDM else remaining_images
        timeout_files = []
        processed_count = 0

        for image_path in iterator:
            try:
                result = analyze_image_enhanced(
                    str(image_path), model, use_cot, cot_depth, 
                    api_type, vllm_url, timeout=timeout
                )
                
                if result is None:
                    if USE_TQDM:
                        tqdm.write(f"Timeout/skip detected for {image_path.name}, adding to timeout list")
                    else:
                        print(f"Timeout/skip detected for {image_path.name}, adding to timeout list")
                    timeout_files.append(str(image_path.name))
                    
                    if timeout_files:
                        with open('timeout_files.txt', 'w') as f:
                            for timeout_file in timeout_files:
                                f.write(f"{timeout_file}\n")
                    continue

                flat_result = flatten_analysis_enhanced(result, str(image_path.name))
                writer.writerow(flat_result)
                processed_count += 1
                
                save_every = 10 if USE_TQDM else 1
                if processed_count % save_every == 0:
                    csvfile.flush()
                    
                    if USE_TQDM:
                        tqdm.write(f"Saved progress: {processed_count} processed, {len(timeout_files)} timeouts")
                    else:
                        print(f"Saved progress: {processed_count} processed, {len(timeout_files)} timeouts")
                
            except Exception as e:
                if USE_TQDM:
                    tqdm.write(f"Error processing {image_path}: {e}")
                else:
                    print(f"Error processing {image_path}: {e}")
                empty_result = flatten_analysis_enhanced(create_empty_analysis(), str(image_path.name))
                writer.writerow(empty_result)
                processed_count += 1
            
            if "vllm" in str(model):
                time.sleep(3)
            else:
                time.sleep(1)

        if timeout_files:
            with open('timeout_files.txt', 'w') as f:
                for timeout_file in timeout_files:
                    f.write(f"{timeout_file}\n")
            print(f"Final summary: {processed_count} images processed, {len(timeout_files)} timeouts saved to timeout_files.txt")

    if USE_TQDM:
        tqdm.write(f"Analysis complete! Results saved to {output_csv}")
    else:
        print(f"Analysis complete! Results saved to {output_csv}")

def check_api_availability(api_type: str, model: str = None, vllm_url: str = "http://localhost:8001") -> bool:
    """Check if specified API and model are available."""
    try:
        if api_type == "ollama":
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            models = [m['name'] for m in response.json().get('models', [])]
            
            if model and model not in models:
                print(f"Warning: Model {model} not found. Available models: {models}")
                return False
            return True
            
        elif api_type == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
            
        elif api_type == "vllm":
            response = requests.get(f"{vllm_url}/health")
            return response.status_code == 200
            
        return False
        
    except Exception as e:
        print(f"Error checking API availability: {e}")
        return False

def main():
    """Main function for image analysis script."""
    args = parse_args()
    
    global USE_TQDM
    if args.no_progress_bar:
        USE_TQDM = False
    
    if not check_api_availability(args.api, args.model, args.vllm_url):
        print(f"Error: API {args.api} with model {args.model} not available")
        return
        
    print(f"✓ Using model: {args.model}")
    
    process_images_enhanced(
        image_folder=args.image_folder,
        output_csv=args.output_csv,
        api_type=args.api,
        model=args.model,
        use_cot=args.use_cot,
        cot_depth=args.cot_depth,
        vllm_url=args.vllm_url,
        sample_size=args.sample_size,
        timeout=args.timeout
    )

if __name__ == "__main__":
    main()