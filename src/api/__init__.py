"""Backend HTTP API for the BuildGuard RAG service.

Exposes the query-to-answer RAG pipeline as a network service so a frontend or
another backend can POST a natural-language question and receive a grounded
answer with structured source citations as JSON.
"""
