import json
import os
import sys
import warnings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Suppress annoying Google API FutureWarnings that flood the terminal
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import google.generativeai as genai
except ImportError:
    print("Please install the Gemini API library: pip install google-generativeai")
    sys.exit(1)

def summarize_events(json_path):
    print(f"Loading and summarizing {json_path} into an optimized context...")
    with open(json_path, 'r') as f:
        events = json.load(f)
    
    summary = {}
    for ev in events:
        pid = str(ev.get('id'))
        raw_class = ev.get('class', 'unknown')
        
        if pid not in summary:
            summary[pid] = {
                'class': raw_class.replace('_U', ' carrying umbrella').replace('_u', ' carrying umbrella'), 
                'color': ev.get('color'),
                'actions': {}, 
                'route_events': set(),
                'first_frame': ev.get('frame'),
                'last_frame': ev.get('frame')
            }
            
        # If the person was later detected holding an umbrella, update their class!
        if '_U' in raw_class or '_u' in raw_class:
            summary[pid]['class'] = 'person carrying umbrella'
        
        action = ev.get('action')
        if action:
            summary[pid]['actions'][action] = summary[pid]['actions'].get(action, 0) + 1
            
        route = ev.get('event')
        if route and "DETECTED" not in route and "STANDING" not in route and "WALKING" not in route and "RUNNING" not in route:
            summary[pid]['route_events'].add(route)
            
        summary[pid]['last_frame'] = max(summary[pid]['last_frame'], ev.get('frame'))
        
    for pid in summary:
        summary[pid]['route_events'] = list(summary[pid]['route_events'])
        
    return summary


def initialize_chat(summary_data, api_key):
    genai.configure(api_key=api_key)
    
    # Google completely swept away the old names on your API tier!
    # We must explicitly request the brand new `gemini-2.5-flash` architecture.
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # We initialize an interactive Chat Session so it remembers follow-up questions!
    prompt = f"Here is the summarized event data from our CCTV video:\n{json.dumps(summary_data, indent=2)}\nYou are an intelligent surveillance AI assistant. Look precisely at the data above and directly answer the questions based on what you see in the data summary."
    
    chat = model.start_chat(history=[
        {"role": "user", "parts": [prompt]},
        {"role": "model", "parts": ["I have analyzed the summarized database and I am fully ready to answer your questions!"]}
    ])
    return chat


if __name__ == "__main__":
    
    # We now fetch the API key from the environment instead of hardcoding it.
    # You can quickly get a free API key at https://aistudio.google.com/app/apikey
    API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        print("Please set your API key beforehand or add it to a .env file if using dotenv.")
        sys.exit(1)
    
    json_file = "events.json"
    if not os.path.exists(json_file):
        print(f"ERROR: {json_file} not found!")
        sys.exit(1)
        
    try:
        summary = summarize_events(json_file)
        chat_session = initialize_chat(summary, API_KEY)
    except Exception as e:
        print(f"Failed to initialize AI: {e}")
        sys.exit(1)
    
    print("\n================ SYSTEM READY ================")
    print("You can now casually ask questions about the video!")
    print("Type 'exit' or press Ctrl+C to stop.")
    print("==============================================\n")
    
    while True:
        try:
            query = input("\nYour Question: ")
            if query.lower() in ['exit', 'quit', 'stop']:
                print("Exiting query engine...")
                break
                
            if not query.strip():
                continue
                
            print("Thinking...")
            response = chat_session.send_message(query)
            
            print("\n>> AI:", response.text)
            
        except KeyboardInterrupt:
            print("\nExiting query engine...")
            break
        except Exception as e:
            print(f"\n[ERROR] Gemini failed to answer: {e}")
            if "API key" in str(e) or "403" in str(e) or "API_KEY_INVALID" in str(e):
                print("\n-> HINT: It looks like your API key is invalid. Make sure it starts with 'AIza...'. You can get one for free at aistudio.google.com/app/apikey")
