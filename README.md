# CodeAlpha_Chatbots-for-FAQ


import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

FAQS = [
    {
        "question": "What is your return policy?",
        "answer": (
            "You can return any item within 30 days of purchase for a full refund. "
            "Items must be unused and in original packaging. "
            "Contact support@shop.com to initiate a return."
        ),
    },
    {
        "question": "How do I track my order?",
        "answer": (
            "Once your order ships, you will receive an email with a tracking number. "
            "Visit our Order Tracking page and enter the number to see real-time updates."
        ),
    },
    {
        "question": "Do you offer free shipping?",
        "answer": (
            "Yes! We offer free standard shipping on all orders over $50. "
            "Orders under $50 have a flat shipping fee of $5.99."
        ),
    },
    {
        "question": "How long does delivery take?",
        "answer": (
            "Standard shipping takes 5–7 business days. "
            "Express (2–3 days) and overnight options are available at checkout."
        ),
    },
    {
        "question": "Can I change or cancel my order?",
        "answer": (
            "You can cancel or modify your order within 1 hour of placing it. "
            "After that it enters processing and cannot be changed. "
            "Email support@shop.com immediately."
        ),
    },
    {
        "question": "What payment methods do you accept?",
        "answer": (
            "We accept Visa, MasterCard, American Express, PayPal, Apple Pay, and Google Pay. "
            "All transactions are secured with SSL encryption."
        ),
    },
    {
        "question": "Is my personal information safe?",
        "answer": (
            "Yes. We use AES-256 encryption to protect your data and never sell your "
            "personal information to third parties. See our Privacy Policy for details."
        ),
    },
    {
        "question": "How do I reset my password?",
        "answer": (
            "Click 'Forgot Password' on the login page, enter your email, and we will "
            "send a reset link valid for 24 hours. Check your spam folder if you don't see it."
        ),
    },
    {
        "question": "Do you ship internationally?",
        "answer": (
            "Yes, we ship to over 50 countries. International delivery takes 10–15 business "
            "days. Customs duties and taxes may apply depending on your country."
        ),
    },
    {
        "question": "How do I contact customer support?",
        "answer": (
            "Email us at support@shop.com, call 1-800-SHOP-NOW (Mon–Fri 9am–6pm EST), "
            "or use the live chat on our website."
        ),
    },
    {
        "question": "What is your warranty policy?",
        "answer": (
            "All products come with a 1-year manufacturer warranty covering defects in "
            "materials and workmanship. Extended warranties are available at checkout."
        ),
    },
    {
        "question": "Can I exchange an item?",
        "answer": (
            "Yes, exchanges are accepted within 30 days. Initiate a return and place "
            "a new order for the desired product; we will expedite your new shipment."
        ),
    },
    {
        "question": "Do you have a loyalty rewards program?",
        "answer": (
            "Yes! Our Rewards Club gives you 1 point per $1 spent. "
            "100 points = $1 off future purchases. Sign up free in your account settings."
        ),
    },
    {
        "question": "Are there any discount codes available?",
        "answer": (
            "Subscribe to our newsletter for a 10% welcome discount. "
            "Follow us on social media for seasonal sales and promo codes."
        ),
    },
    {
        "question": "How do I create an account?",
        "answer": (
            "Click 'Sign Up' at the top right, enter your name, email, and password. "
            "Confirm the verification email to activate your account instantly."
        ),
    },
]


class NLPPreprocessor:
    """Tokenize, clean, remove stopwords, and lemmatize text."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:

        text = text.lower()

        text = re.sub(r"[^a-z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return " ".join(tokens)


class FAQChatbot:
    """
    Matches a user query to the most similar FAQ using TF-IDF + cosine similarity.
    """

    CONFIDENCE_THRESHOLD = 0.10

    def __init__(self, faqs: list[dict]):
        self.faqs = faqs
        self.preprocessor = NLPPreprocessor()

        self.raw_texts = [f["question"] + " " + f["answer"] for f in faqs]
        self.processed_texts = [self.preprocessor.preprocess(t) for t in self.raw_texts]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_texts)

        print(" FAQ Chatbot initialised — {} FAQs loaded.\n".format(len(self.faqs)))

    def get_best_match(self, user_query: str) -> dict:
        processed_query = self.preprocessor.preprocess(user_query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        return {
            "matched_question": self.faqs[best_idx]["question"],
            "answer": self.faqs[best_idx]["answer"],
            "confidence": best_score,
            "index": best_idx,
        }

    def get_top_matches(self, user_query: str, top_n: int = 3) -> list[dict]:
        processed_query = self.preprocessor.preprocess(user_query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_n]
        return [
            {
                "question": self.faqs[i]["question"],
                "answer": self.faqs[i]["answer"],
                "confidence": float(scores[i]),
            }
            for i in top_indices
        ]

    def respond(self, user_query: str) -> str:
        if not user_query.strip():
            return "Please enter a question."

        result = self.get_best_match(user_query)

        if result["confidence"] < self.CONFIDENCE_THRESHOLD:
            return (
                "⚠️  I'm not confident I have an answer for that.\n"
                "Try rephrasing, or contact us at support@shop.com."
            )

        confidence_pct = round(result["confidence"] * 100, 1)
        return (
            f" Matched FAQ : {result['matched_question']}\n"
            f" Confidence  : {confidence_pct}%\n"
            f"\n Answer:\n{result['answer']}"
        )

    def show_preprocessing(self, text: str):
        print("\n── NLP Preprocessing Pipeline ──────────────────")
        print(f"  Original  : {text}")
        lower   = text.lower()
        print(f"  Lowercase : {lower}")
        cleaned = re.sub(r"[^a-z\s]", "", lower)
        print(f"  Cleaned   : {cleaned}")
        tokens  = word_tokenize(cleaned)
        print(f"  Tokens    : {tokens}")
        sw      = set(stopwords.words("english"))
        filtered= [t for t in tokens if t not in sw and len(t) > 2]
        print(f"  No stops  : {filtered}")
        lem     = WordNetLemmatizer()
        lemmas  = [lem.lemmatize(t) for t in filtered]
        print(f"  Lemmatized: {lemmas}")
        print(f"  Final str : {' '.join(lemmas)}")
        print("────────────────────────────────────────────────\n")

    def chat(self):
        print("=" * 56)
        print("       🤖  FAQ Chatbot  (type 'quit' to exit)")
        print("         Commands:  'top3' · 'preprocess' · 'list'")
        print("=" * 56)

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye! 👋")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit", "bye"):
                print("Bot: Goodbye! Have a great day! 👋")
                break

            elif cmd == "list":
                print("\nBot: Here are all available FAQs:\n")
                for i, faq in enumerate(self.faqs, 1):
                    print(f"  {i:>2}. {faq['question']}")

            elif cmd.startswith("preprocess"):

                text = user_input[len("preprocess"):].strip() or "What is your return policy?"
                self.show_preprocessing(text)

            elif cmd.startswith("top3"):
                query = user_input[4:].strip()
                if not query:
                    print("Bot: Usage → top3 <your question>")
                    continue
                matches = self.get_top_matches(query)
                print("\nBot: Top 3 matches:\n")
                for rank, m in enumerate(matches, 1):
                    pct = round(m["confidence"] * 100, 1)
                    print(f"  {rank}. [{pct}%] {m['question']}")

            else:
                response = self.respond(user_input)
                print(f"\nBot: {response}")

def run_demo(bot: FAQChatbot):
    demo_queries = [
        "How can I send back a product I bought?",
        "Where is my package right now?",
        "Is delivery free?",
        "I forgot my password, what should I do?",
        "Do you deliver to other countries?",
        "What credit cards do you take?",
        "Can I earn points for shopping?",
        "How do I get in touch with you?",
        "Tell me about the warranty",
        "How to make a new account?",
    ]

    print("\n" + "=" * 56)
    print("            DEMO — Automated Test Queries")
    print("=" * 56)

    for query in demo_queries:
        print(f"\n User : {query}")
        result = bot.get_best_match(query)
        pct = round(result["confidence"] * 100, 1)
        print(f"   Match: {result['matched_question']} [{pct}%]")

        answer = result["answer"]
        print(f"   Reply: {answer[:90]}{'…' if len(answer) > 90 else ''}")

    print("\n" + "=" * 56)

    bot.show_preprocessing("How can I send back a product I bought?")


if __name__ == "__main__":
    bot = FAQChatbot(FAQS)

    import sys
    if "--demo" in sys.argv:
        run_demo(bot)
    else:

        run_demo(bot)
        bot.chat()
