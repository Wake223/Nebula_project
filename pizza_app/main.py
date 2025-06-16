from google import generativeai as genai# Changed import for simpler model access
from collections import deque
import uuid
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Memory Component ---
class SimpleMemory:
    def __init__(self, max_messages=10):
        self.messages = deque(maxlen=max_messages)

    def add_message(self, role, content):
        self.messages.append(f"{role}: {content}")

    def get_history(self):
        return "\n".join(self.messages)

    def clear(self):
        self.messages.clear()

# --- Order Management Component ---
class MenuItem:
    def __init__(self, name: str, size: str, price: float, quantity: int = 1):
        self.name = name
        self.size = size
        self.price = price
        self.quantity = quantity

    def __str__(self):
        return f"{self.quantity}x {self.size.capitalize()} {self.name.capitalize()} (${self.price * self.quantity:.2f})"

class PizzaOrderManager:
    def __init__(self):
        self.order_id = str(uuid.uuid4())
        self.items = [] # List of MenuItem objects
        self.customer_info = {"name": "", "phone": "", "address": ""}
        self.total_price = 0.0

        self.menu = {
            "pepperoni": {"small": 10.00, "medium": 14.00, "large": 18.00},
            "margherita": {"small": 9.00, "medium": 13.00, "large": 17.00},
            "vegetarian": {"small": 11.00, "medium": 15.00, "large": 19.00},
            "bbq chicken": {"small": 12.00, "medium": 16.00, "large": 20.00},
            "coke": {"regular": 2.50},
            "sprite": {"regular": 2.50}
        }

    def add_item(self, item_name: str, size: str, quantity: int = 1) -> bool:
        item_name_lower = item_name.lower()
        size_lower = size.lower()

        if item_name_lower in self.menu and size_lower in self.menu[item_name_lower]:
            price_per_unit = self.menu[item_name_lower][size_lower]
            for item in self.items:
                if item.name == item_name_lower and item.size == size_lower:
                    item.quantity += quantity
                    self._calculate_total()
                    return True
            self.items.append(MenuItem(item_name_lower, size_lower, price_per_unit, quantity))
            self._calculate_total()
            return True
        return False

    def set_customer_info(self, name: str = "", phone: str = "", address: str = ""):
        if name: self.customer_info["name"] = name
        if phone: self.customer_info["phone"] = phone
        if address: self.customer_info["address"] = address

    def _calculate_total(self):
        self.total_price = sum(item.price * item.quantity for item in self.items)

    def get_order_summary(self) -> str:
        if not self.items:
            return "Your order is currently empty."

        summary_lines = ["Current Order:"]
        for item in self.items:
            summary_lines.append(f"- {item}")
        summary_lines.append(f"Total: ${self.total_price:.2f}")

        if self.customer_info["name"] or self.customer_info["address"] or self.customer_info["phone"]:
            delivery_details = []
            if self.customer_info["name"]: delivery_details.append(f"Name: {self.customer_info['name']}")
            if self.customer_info["address"]: delivery_details.append(f"Address: {self.customer_info['address']}")
            if self.customer_info["phone"]: delivery_details.append(f"Phone: {self.customer_info['phone']}")
            summary_lines.append(f"Delivery Info: {', '.join(delivery_details)}.")

        return "\n".join(summary_lines)

    def clear_order(self):
        self.items = []
        self.customer_info = {"name": "", "phone": "", "address": ""}
        self.total_price = 0.0
        self.order_id = str(uuid.uuid4())

    def get_menu_items(self) -> str:
        menu_str = "Our Delicious Pizza Nebula Menu:\n"
        for item, sizes in self.menu.items():
            if item not in ["coke", "sprite"]:
                menu_str += f"- {item.capitalize()}:\n"
                for size, price in sizes.items():
                    menu_str += f"    {size.capitalize()}: ${price:.2f}\n"
        menu_str += "\nDrinks:\n"
        for item, sizes in self.menu.items():
            if item in ["coke", "sprite"]:
                for size, price in sizes.items():
                    menu_str += f"- {item.capitalize()} ({size.capitalize()}): ${price:.2f}\n"
        return menu_str

# --- Main Chatbot Class ---
class PizzaNebulaChatBot:
    def __init__(self, api_key: str, max_memory_messages: int = 20):
        # Configure the genai library with the API key
        genai.configure(api_key=api_key)

        # Initialize the GenerativeModel directly
        self.intent_model = genai.GenerativeModel("gemini-1.5-flash-latest") # Using 1.5 for potentially better JSON
        self.chat_model = genai.GenerativeModel("gemini-1.5-flash-latest") # Using 1.5 for general chat

        self.memory = SimpleMemory(max_messages=max_memory_messages)
        self.order_manager = PizzaOrderManager()

        self.general_system_prompt = """You are Pizza Nebula, a friendly and efficient pizza restaurant employee.
Your goal is to help customers order pizza, answer questions about the menu, and confirm delivery details.
Keep responses concise, helpful, and friendly. Always refer to the customer's current order when appropriate.
You are focusing on customer interaction and dialogue. Do not invent details not provided by the user or the given context.
"""
        self.intent_extraction_prompt = """You are an intent recognition and entity extraction system for a pizza ordering chatbot.
Analyze the user's input and identify their primary intent and any relevant entities.

**Available Intents:**
- **ORDER_ITEM**: User wants to order a pizza or drink.
  * Entities: `item` (string, e.g., "pepperoni", "coke"), `size` (string, e.g., "small", "medium", "large", "regular"), `quantity` (integer, default 1 if not specified, only positive numbers).
- **GET_MENU**: User wants to see the menu.
- **GET_ORDER_SUMMARY**: User wants to know what they've ordered so far.
- **PROVIDE_DELIVERY_INFO**: User is giving their name, phone number, or address. Extract all available info.
  * Entities: `name` (string), `phone` (string, numbers only), `address` (string).
- **CONFIRM_ORDER**: User is confirming a previous step or order.
- **DECLINE_ORDER**: User is declining a previous step or modification.
- **OFF_TOPIC**: User is talking about something completely unrelated to pizza ordering.
- **GOODBYE**: User wants to end the conversation.
- **GREETING**: User is just saying hello.
- **UNKNOWN**: If you cannot determine a clear intent from the above, or the input is ambiguous.

**Output Format:**
Respond *only* with a JSON object. The JSON object must contain an "intent" key and an "entities" key.
The "entities" value should be a JSON object containing extracted information. If no entities are found for an intent, the "entities" object can be empty (`{}`).

**Examples:**
User: "I want a large pepperoni pizza"
JSON: `{"intent": "ORDER_ITEM", "entities": {"item": "pepperoni", "size": "large", "quantity": 1}}`

User: "can I see the menu"
JSON: `{"intent": "GET_MENU", "entities": {}}`

User: "my name is Alex and my address is 123 Main St"
JSON: `{"intent": "PROVIDE_DELIVERY_INFO", "entities": {"name": "Alex", "address": "123 Main St"}}`

User: "What's the weather like?"
JSON: `{"intent": "OFF_TOPIC", "entities": {}}`

User: "Bye"
JSON: `{"intent": "GOODBYE", "entities": {}}`

User: "What have I ordered?"
JSON: `{"intent": "GET_ORDER_SUMMARY", "entities": {}}`

User: "Hello there!"
JSON: `{"intent": "GREETING", "entities": {}}`

User Input: "{user_input}"
JSON:
"""

        self.off_topic_counter = 0
        self.off_topic_threshold = 3
        self.exit_flag = False

    def _extract_intent_and_entities(self, user_input: str) -> dict:
        """
        Uses the LLM to extract the user's intent and entities in a structured JSON format.
        """
        try:
            # FIX: Use the specific intent_model and pass contents as a list of parts
            response = self.intent_model.generate_content(
                [self.intent_extraction_prompt.format(user_input=user_input)],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0 # Keep temperature low for structured output
                )
            )

            # Debugging: Print raw response from intent extraction
            #print(f"DEBUG: Raw Intent LLM Response: {response.text}")

            json_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
            return json.loads(json_str)
        except (json.JSONDecodeError, Exception) as e:
            #print(f"ERROR: Intent extraction failed or JSON malformed: {e}")
            # Fallback to UNKNOWN intent if parsing fails
            return {"intent": "UNKNOWN", "entities": {}}

    def process_message(self, user_input: str) -> tuple[str, bool]:
        """
        Processes a single user message through intent recognition,
        dialogue management, and response generation.
        """
        self.memory.add_message("Customer", user_input)
        final_bot_response = ""
        should_exit = False

        # --- Step 1: Intent and Entity Extraction ---
        parsed_data = self._extract_intent_and_entities(user_input)
        intent = parsed_data.get("intent", "UNKNOWN")
        entities = parsed_data.get("entities", {})

        #print(f"DEBUG: Recognized Intent: {intent}, Entities: {entities}") # Debugging

        # --- Step 2: Dialogue Management based on Intent ---
        if intent == "OFF_TOPIC":
            self.off_topic_counter += 1
            if self.off_topic_counter >= self.off_topic_threshold:
                final_bot_response = (
                    "It seems we're not discussing pizza. I'm Pizza Nebula, "
                    "here to help with your pizza order. If you change your mind, "
                    "feel free to start a new order! Goodbye!"
                )
                should_exit = True
            else:
                final_bot_response = "I'm here to help you order pizza! What would you like?"

            # Generate natural language response based on system prompt and specific message
            gen_response = self.chat_model.generate_content(
                [f"{self.general_system_prompt}\n\nCustomer's input: '{user_input}'. Bot's decision: '{final_bot_response}'. Generate a polite redirection or farewell based on this decision."],
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            final_bot_response = gen_response.text

        elif intent == "GOODBYE":
            self.off_topic_counter = 0
            final_bot_response = "Thanks for choosing Pizza Nebula! Goodbye!"
            should_exit = True

        elif intent == "ORDER_ITEM":
            self.off_topic_counter = 0 # Reset counter if on-topic
            item_name = entities.get("item")
            size = entities.get("size")
            quantity = entities.get("quantity", 1) # Default to 1 if not specified

            if item_name and size:
                if self.order_manager.add_item(item_name, size, quantity):
                    summary = self.order_manager.get_order_summary()
                    gen_response = self.chat_model.generate_content(
                        [f"{self.general_system_prompt}\n\nCustomer ordered '{quantity} {size} {item_name}'. Item successfully added to order. Current order details: {summary}. Generate a friendly confirmation message and ask if they need anything else."],
                        generation_config=genai.types.GenerationConfig(temperature=0.7)
                    )
                    final_bot_response = gen_response.text
                else:
                    gen_response = self.chat_model.generate_content(
                        [f"{self.general_system_prompt}\n\nCustomer attempted to order '{quantity} {size} {item_name}'. Item or size not found in menu. Respond apologetically and suggest checking the menu or clarifying based on the menu provided."],
                        generation_config=genai.types.GenerationConfig(temperature=0.7)
                    )
                    final_bot_response = gen_response.text
            else:
                gen_response = self.chat_model.generate_content(
                    [f"{self.general_system_prompt}\n\nCustomer mentioned ordering but details are unclear ('{user_input}'). Ask for clarification on item type and size. Provide the menu again if helpful: {self.order_manager.get_menu_items()}"],
                    generation_config=genai.types.GenerationConfig(temperature=0.7)
                )
                final_bot_response = gen_response.text

        elif intent == "GET_MENU":
            self.off_topic_counter = 0
            menu_text = self.order_manager.get_menu_items()
            gen_response = self.chat_model.generate_content(
                [f"{self.general_system_prompt}\n\nCustomer asked for the menu. Provide this menu: {menu_text}. Then ask what they'd like to order, keeping it friendly and concise."],
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            final_bot_response = gen_response.text

        elif intent == "GET_ORDER_SUMMARY":
            self.off_topic_counter = 0
            summary_text = self.order_manager.get_order_summary()
            gen_response = self.chat_model.generate_content(
                [f"{self.general_system_prompt}\n\nCustomer asked for order summary. Provide this summary: {summary_text}. Then ask if they want to add anything else or finalize their order."],
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            final_bot_response = gen_response.text

        elif intent == "PROVIDE_DELIVERY_INFO":
            self.off_topic_counter = 0
            name = entities.get("name", "")
            phone = entities.get("phone", "")
            address = entities.get("address", "")

            self.order_manager.set_customer_info(name=name, phone=phone, address=address)

            summary_text = self.order_manager.get_order_summary()

            # Check for missing info
            missing_info = []
            if not self.order_manager.customer_info["name"]: missing_info.append("name")
            if not self.order_manager.customer_info["phone"]: missing_info.append("phone number")
            if not self.order_manager.customer_info["address"]: missing_info.append("address")

            if missing_info:
                missing_str = ", ".join(missing_info)
                gen_response = self.chat_model.generate_content(
                    [f"{self.general_system_prompt}\n\nCustomer provided some delivery info. Current details: {summary_text}. Please ask for the remaining missing information: {missing_str}. Do not invent details."],
                    generation_config=genai.types.GenerationConfig(temperature=0.7)
                )
                final_bot_response = gen_response.text
            else:
                gen_response = self.chat_model.generate_content(
                    [f"{self.general_system_prompt}\n\nCustomer provided all delivery info. Current order details: {summary_text}. Ask for final confirmation to proceed with the order."],
                    generation_config=genai.types.GenerationConfig(temperature=0.7)
                )
                final_bot_response = gen_response.text

        elif intent == "GREETING":
            self.off_topic_counter = 0
            gen_response = self.chat_model.generate_content(
                [f"{self.general_system_prompt}\n\nCustomer said hello. Greet them back warmly and ask what they'd like to order today, referencing that you're Pizza Nebula."],
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            final_bot_response = gen_response.text

        elif intent in ["CONFIRM_ORDER", "DECLINE_ORDER"]:
            self.off_topic_counter = 0
            # For confirmation/decline, provide full context from order manager and let LLM generate
            prompt_for_confirmation = self.general_system_prompt + "\n\n"
            current_order_summary = self.order_manager.get_order_summary()
            if "Your order is currently empty." not in current_order_summary:
                prompt_for_confirmation += f"Current Order State: {current_order_summary}\n\n"

            history = self.memory.get_history()
            if history:
                prompt_for_confirmation += history + "\n"

            prompt_for_confirmation += f"Customer: {user_input}\nPizza Employee:"

            gen_response = self.chat_model.generate_content(
                [prompt_for_confirmation],
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            final_bot_response = gen_response.text

        else: # UNKNOWN or other unhandled intents, fall back to general LLM response
            self.off_topic_counter = 0
            # Use the general system prompt with conversation history for broad queries
            prompt_for_general_response = self.general_system_prompt + "\n\n"
            current_order_summary = self.order_manager.get_order_summary()
            if "Your order is currently empty." not in current_order_summary:
                prompt_for_general_response += f"Current Order State: {current_order_summary}\n\n"

            history = self.memory.get_history()
            if history:
                prompt_for_general_response += history + "\n"

            prompt_for_general_response += f"Customer: {user_input}\nPizza Employee:"

            gen_response = self.chat_model.generate_content(
                [prompt_for_general_response],
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            final_bot_response = gen_response.text

        # Add the bot's final response to memory
        self.memory.add_message("Pizza Employee", final_bot_response)
        return final_bot_response, should_exit

    def run(self):
        """Starts the interactive chat loop."""
        print("Welcome to Pizza Nebula!")
        print(f"{self.order_manager.get_menu_items()}\n")
        print("Type 'quit', 'exit' or 'bye' to exit the chat.\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thanks for choosing Pizza Nebula! Goodbye!")
                break

            if not user_input:
                continue

            bot_response, should_exit = self.process_message(user_input)
            print(f"\nPizza Bot: {bot_response}\n")

            if should_exit:
                print("Ending the chat as requested by Pizza Nebula. Have a great day!")
                break

if __name__ == "__main__":
    pizza_bot = PizzaNebulaChatBot(api_key=API_KEY)
    pizza_bot.run()