import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import random  # For varied farewell messages
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)


class VectorBookStore:
    """
    Manages a book store with vector-based search capabilities using TF-IDF
    and integrates with a Generative AI model for conversational responses.
    """

    def __init__(self, csv_file: str):
        """
        Initializes the VectorBookStore with book data from a CSV file.

        Args:
            csv_file (str): The path to the CSV file containing book data.
        """
        self.df = pd.read_csv(csv_file)
        # Dictionaries to store separate vectorizers and vectors for each column
        self.column_vectorizers = {}
        self.column_vectors = {}
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.setup_vectors()

    def setup_vectors(self):
        """
        Prepares the vector database by creating separate TF-IDF vectors for
        each relevant text-based column (title, authors, description, categories).
        Handles missing data and ensures data types are strings.
        """
        # Columns to be vectorized individually
        vectorize_columns = ['title', 'authors', 'description', 'categories']

        # Ensure all necessary columns exist and fill NaNs
        for col in vectorize_columns:
            if col not in self.df.columns:
                print(f"Warning: Column '{col}' not found in CSV. Skipping vectorization for this column.")
                continue
            self.df[col] = self.df[col].fillna('').astype(str)

            # Initialize a TfidfVectorizer for each column
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            # Fit and transform the column data
            vectors = vectorizer.fit_transform(self.df[col])

            self.column_vectorizers[col] = vectorizer
            self.column_vectors[col] = vectors
            #print(f"📊 Vector index created for '{col}' with {vectors.shape[0]} documents and {vectors.shape[1]} features.")

        # Also handle 'published_year' and 'average_rating' which are not vectorized but used in context
        if 'published_year' in self.df.columns:
            self.df['published_year'] = self.df['published_year'].fillna('').astype(str)
        if 'average_rating' in self.df.columns:
            self.df['average_rating'] = self.df['average_rating'].fillna(0.0).astype(
                float)  # Fill with 0.0 for numeric operations

        #print(f"✅ Vector database ready for {len(self.df)} books.")

    def search_books(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Performs a vector-based search across multiple columns (title, authors, description, categories).
        Combines similarities from each column to find the most relevant books.

        Args:
            query (str): The user's search query.
            top_k (int): The number of top relevant books to return.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents a
                        relevant book with its details and combined similarity score.
        """
        combined_similarities = None

        for col_name, vectorizer in self.column_vectorizers.items():
            query_vector = vectorizer.transform([query])
            # Calculate similarity for the current column
            column_similarities = cosine_similarity(query_vector, self.column_vectors[col_name]).flatten()

            if combined_similarities is None:
                combined_similarities = column_similarities
            else:
                # Add similarities from each column. You could apply weights here if desired.
                combined_similarities += column_similarities

        if combined_similarities is None:
            return []  # No columns were vectorized

        # Get indices of top_k most similar books
        top_indices = combined_similarities.argsort()[-top_k:][::-1]

        results = []
        # Define a threshold for similarity to filter out very low matches
        similarity_threshold = 0.1

        for idx in top_indices:
            # Check if the combined similarity is above the threshold
            if combined_similarities[idx] > similarity_threshold:
                book = self.df.iloc[idx].to_dict()
                book['combined_similarity'] = combined_similarities[idx]
                results.append(book)
        return results

    def chat_with_llm(self, user_message: str, relevant_books: list[dict] = None) -> str:
        """
        Interacts with the Generative AI model to provide a conversational response.
        Optionally includes context from relevant books found by the search function.

        Args:
            user_message (str): The user's input message.
            relevant_books (list[dict], optional): A list of relevant books to
                                                    provide as context to the LLM. Defaults to None.

        Returns:
            str: The AI model's response in English.
        """
        context = ""
        if relevant_books:
            # Format relevant books for the LLM context
            context_books_info = []
            for book in relevant_books:
                book_info = {
                    "title": book.get('title', 'N/A'),
                    "authors": book.get('authors', 'N/A'),
                    "description": book.get('description', 'N/A'),
                    "categories": book.get('categories', 'N/A'),
                    "published_year": book.get('published_year', 'N/A'),
                    "average_rating": book.get('average_rating', 'N/A')
                }
                context_books_info.append(book_info)
            context = f"\nRelevant books from the database:\n{json.dumps(context_books_info, ensure_ascii=False, indent=1)}\n"

        # English prompt for the LLM with instructions for staying on topic and responding in English
        prompt = f"""You are a friendly and helpful assistant for a bookstore. You have access to a large database of books.

User message: "{user_message}"
{context}

Instructions:
- Be natural and friendly.
- **Respond in English.**
- If the question is about books, use the provided book data to give concrete and useful recommendations.
- If it's a general conversation, respond warmly but **always redirect the user back to discussing books or how you can help them with book selection.**
- Do not mention technical details.
- Provide specific and helpful advice regarding books.
- Try to keep answers concise and to the point, yet informative.
- If a book is not found, but the user is asking about a specific book, you can state that the book was not found in the available data but offer to help find similar books.
- **Crucially, if the user tries to talk about topics unrelated to books, gently remind them that you are a bookstore assistant and can only help with book-related queries, then offer assistance with book selection.**
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I am unable to respond at the moment. Error: {e}"


def main():
    """
    Main function to run the VectorBookStore application.
    Handles user input, book search, LLM interaction, and application termination.
    """
    print("📚 Welcome to the Bookstore!")

    # Automatically looks for 'data.csv' in the same directory
    csv_file = "data 2.csv"

    # List of varied farewell messages
    farewell_messages = [
        "👋 Goodbye! Have a great day!",
        "✨ See you soon! Take care!",
        "📚 Happy reading! Farewell!",
        "😊 Goodbye! Hope to see you again!",
        "👋 Thanks for visiting! All the best!",
        "🌸 Farewell! Hope to assist you again!"
    ]

    try:
        store = VectorBookStore(csv_file)
        #print("✅ Book database loaded! You can now ask any question.\n")

        while True:
            user_input = input("👤 ").strip()

            if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye', 'see you', 'dont want nothing']:
                print(random.choice(farewell_messages))
                break

            if not user_input:
                continue

            # Always search for relevant books based on user input
            relevant_books = store.search_books(user_input, top_k=5)

            # Get LLM response
            response = store.chat_with_llm(user_input, relevant_books)
            print(f"🤖 {response}\n")

    except FileNotFoundError:
        print(f"❌ Error: The file '{csv_file}' was not found!")
        print("Please make sure that 'data.csv' is in the same folder as the program.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
