#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Client, uses OpenAI Python library to call LLM
Supports configurable base_url (defaults to DeepSeek)
"""

import os
import time
import random
from typing import Optional, Dict, Any, List
from openai import OpenAI


class LLMClient:
    """LLM client class, wraps OpenAI API calls"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: int = 300,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        retry_backoff_factor: float = 2.0,
        retry_jitter: bool = True,
        reasoning_effort: Optional[str] = None
    ):
        """
        Initialize LLM client

        Args:
            api_key: API key, reads from OPENAI_API_KEY environment variable if None
            base_url: API base URL, defaults to DeepSeek
            model: Model name, defaults to deepseek-chat
            timeout: Request timeout (seconds)
            max_retries: Maximum number of retries
            retry_base_delay: Base delay time (seconds)
            retry_max_delay: Maximum delay time (seconds)
            retry_backoff_factor: Backoff factor
            retry_jitter: Whether to add random jitter
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_jitter = retry_jitter
        self.reasoning_effort = reasoning_effort

        if not self.api_key:
            raise ValueError("API key not provided, please set api_key parameter or OPENAI_API_KEY environment variable")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text content

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Temperature parameter, controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other OpenAI API parameters

        Returns:
            Generated text content
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

        if max_tokens:
            params["max_tokens"] = max_tokens
        
        return self._generate_with_retry(
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            backoff_factor=self.retry_backoff_factor,
            jitter=self.retry_jitter,
            **params
        )

    def _generate_with_retry(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        **params
    ) -> str:
        """
        Generation method with retry mechanism

        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay time (seconds)
            max_delay: Maximum delay time (seconds)
            backoff_factor: Backoff factor
            jitter: Whether to add random jitter
            **params: API parameters

        Returns:
            Generated text content

        Raises:
            Exception: Raises exception after retries exhausted
        """
        last_exception = None

        for attempt in range(max_retries + 1):  # +1 because first attempt doesn't count as retry
            try:
                response = self.client.chat.completions.create(**params)

                time.sleep(random.uniform(0.5, 1.5))
                return response.choices[0].message.content

            except Exception as e:
                last_exception = e

                if attempt < max_retries:
                    # Calculate delay time
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                    # Add random jitter to avoid thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    print(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    print(f"Retrying in {delay:.1f} seconds...")

                    time.sleep(delay)
                else:
                    print(f"LLM call retry failed, maximum retries {max_retries} reached")

        # All retries failed
        raise Exception(f"LLM call failed after {max_retries} retries. Last error: {str(last_exception)}")
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Stream generate text content (generator)

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Temperature parameter, controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other OpenAI API parameters

        Yields:
            Generated text fragments
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        try:
            stream = self.client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"LLM stream call failed: {str(e)}")
    
    def update_base_url(self, base_url: str):
        """Update base_url"""
        self.base_url = base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def update_model(self, model: str):
        """Update model name"""
        self.model = model


# Convenience function: create default client
def create_client(
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    reasoning_effort: Optional[str] = None
) -> LLMClient:
    """
    Create default LLM client

    Args:
        api_key: API key
        base_url: API base URL, defaults to DeepSeek
        model: Model name, defaults to deepseek-chat

    Returns:
        LLMClient instance
    """
    return LLMClient(api_key=api_key, base_url=base_url, model=model)


# Example usage
if __name__ == "__main__":
    # Method 1: Use default configuration (DeepSeek)
    # client = create_client(api_key="your-api-key")
    

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('BASE_URL')
    
    str = '''Please introduce the history of artificial intelligence'''
    client = create_client(api_key=api_key, base_url=base_url)
    result = client.generate(
        model="gemini-2.5-pro",
        prompt=str,
        temperature=0.7
    )
    print(result)

    # Method 2: Custom base_url (e.g., using OpenAI)
    # client = create_client(
    #     api_key="your-api-key",
    #     base_url="https://api.openai.com/v1",
    #     model="gpt-4"
    # )

    # Method 3: Use API key from environment variable
    # export OPENAI_API_KEY="your-api-key"
    # client = create_client()

    # Generate content
    # result = client.generate(
    #     prompt="Please introduce the history of artificial intelligence",
    #     system_prompt="You are a professional technology consultant",
    #     temperature=0.7
    # )
    # print(result)

    # Stream generation
    # for chunk in client.generate_stream(prompt="Write a poem about spring"):
    #     print(chunk, end="", flush=True)

