"""
Curated Benign Prompt Corpus for Contrastive Security Scoring in ARES Gateway.

Contains diverse non-malicious user queries across software engineering,
data processing, security research, system administration, and conversational tasks.
Used to compute contrastive margin: Margin = Sim(Attack) - Sim(Benign).
"""

from typing import List

BENIGN_REFERENCE_PROMPTS: List[str] = [
    # Software Engineering & Programming Queries (often sharing keywords with encoding/delimiters)
    "How do I decode a base64 string in Python?",
    "Can you show me how to encode and decode hex strings in JavaScript?",
    "Explain how ROT13 cipher works and give an example in C#.",
    "How do I parse XML documents and handle nested tags in Java?",
    "How do I format JSON responses using Pydantic in FastAPI?",
    "What is the difference between single quotes and triple quotes in Python strings?",
    "How do I sanitize HTML input to prevent cross-site scripting (XSS)?",
    "Can you help me write a regular expression to match email addresses?",
    "How to use Markdown code blocks and syntax highlighting in GitHub README?",
    "Show me a SQL query with JOIN and GROUP BY clauses to aggregate customer orders.",
    "How do I handle exceptions and print error tracebacks in Python?",
    "Can you explain the difference between base64 and standard ASCII encoding?",
    "What is the best way to serialize binary data in Node.js?",
    "How do I escape special characters like < and > in HTML templates?",
    "Write a C# method to validate GUID strings and parse JSON payloads.",

    # System Administration & DevOps (often sharing keywords like system, admin, root)
    "How do I check system resource usage on Ubuntu Linux?",
    "Explain the difference between root user and a sudo user in Linux.",
    "What does chmod 755 do and how do file permissions work in Unix?",
    "How do I configure environment variables in Docker Compose?",
    "How do I inspect system logs using journalctl on systemd?",
    "What is the recommended way to restart an nginx service gracefully?",
    "How to secure SSH server access using public key authentication?",
    "Can you explain what systemd units and service timers are?",
    "How do I monitor CPU and memory utilization in Windows Server?",
    "What is the best way to manage database connection pools in production?",

    # AI & Prompt Engineering Knowledge (often sharing keywords like system prompt, instructions)
    "What is a system prompt and how does it shape model persona?",
    "How can developers write clear instructions for technical documentation bots?",
    "Explain the role of temperature and top_p in language model generation.",
    "What are few-shot prompting techniques and when should they be used?",
    "How do zero-trust security boundaries work in AI application architecture?",
    "What is the purpose of canary tokens in automated prompt security testing?",
    "Can you explain what the OWASP Top 10 for Large Language Models covers?",
    "How do developers implement guardrails to prevent hallucination in LLMs?",
    "What is the difference between fine-tuning and retrieval-augmented generation?",
    "How do I evaluate prompt robustness against adversarial input in production?",

    # Role-Play & Creative Writing (benign creative contexts)
    "Pretend you are a medieval historian explaining the Magna Carta to high school students.",
    "Act as an interviewer asking behavioral questions for a senior software engineer position.",
    "Can you write a poem about autumn leaves falling in a peaceful mountain valley?",
    "Imagine you are a customer service representative helping a user track an order.",
    "Help me draft a polite email requesting an extension on a project deadline.",
    "Roleplay as a tour guide giving a tour of ancient Rome's Colosseum.",
    "Write a fictional dialogue between two scientists discussing renewable energy.",
    "Act as a writing coach reviewing my opening paragraph for clarity and tone.",

    # Information Retrieval & Knowledge Assistance
    "What is the capital of Australia and what is its population?",
    "Can you summarize the main causes of the Industrial Revolution?",
    "How does photosynthesis convert sunlight into chemical energy in plants?",
    "Explain the difference between supervised and unsupervised machine learning.",
    "What are the primary differences between relational and NoSQL databases?",
    "Can you explain how HTTPS certificates and TLS handshakes provide encryption?",
    "Summarize the key principles of the Agile software development methodology.",
    "What is the Doppler effect and where is it observed in astronomy?",
]
