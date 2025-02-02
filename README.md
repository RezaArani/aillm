# **Golang RAG/LLM framework with Memory and Transcriber - All-in-One Platform**
  

This project is an easy to use **RAG (Retrieval-Augmented Generation)** system that combines an **LLM client with memory**, **text embedding**, and **document transcription** in a single easy-to-configure platform  framework over [LangChain Go](https://github.com/tmc/langchaingo).

It offers seamless **compatibility with OLLAMA and OpenAI**, making it an excellent choice for chatbot applications.

  

## **Features**

  

1.  **LLM Client - Sync and Stream with High Compatibility**

- Supports both synchronous and streaming interactions with LLMs.

- Fully compatible with **OLLAMA** and **OpenAI**, ensuring smooth chatbot integration.

  

2.  **Text Embedding & Similarity Search**

- Efficient document embedding and retrieval using **Cosine Similarity** and **K-Nearest Neighbors (KNN)**.

- Powered by **Redis Vector Store** for fast lookups.

  


3.  **Document Transcription**
- Internet content downloader
- Processes files such as PDFs, Word, Excel, and HTML using **Apache Tika**.

- Extracts structured data, including OCR-based transcription for scanned documents.
  

4.  **Multilingual Support**

- Process and analyze content in multiple languages with AI-powered detection and response customization.

  

5.  **Scalability**

- Leverages Redis and efficient text processing pipelines to scale with large datasets.

  

## **Configuration**

  


  

Configure environment variables in `.env` file:

  

```env

APITOKEN=your-openai-api-token //not mandatory for Ollama

REDIS_HOST=localhost:6379

REDIS_PASSWORD=your_redis_password

TIKA_URL=http://localhost:9998/tika //not mandatory for text and html usage

```

  

## **Usage**

  
Refer to the [examples](https://github.com/RezaArani/aillm/tree/master/examples) folder in the repository for more details. Below is a simple usage example:



  

```go
package main

import (
	"context"
	"fmt"
	"log"

	aillm "github.com/RezaArani/aillm/controller"
)

func main() {
	log.Println("Testing aillm framework:")
	// locally hosted ollama
	llmclient := &aillm.OllamaController{
		Config: aillm.LLMConfig{
			Apiurl:  "http://127.0.0.1:11434",
			AiModel: "llama3.1",
		},
	}
	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		Embedder:  llmclient,
		LLMClient: llmclient,
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	
	llm.Init()
	// asking question without context
	askKLLM(llm, "What is SemMapas?")
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm)
	
	// Time for asking some questions
	askKLLM(llm, "What is SemMapas?")
	askKLLM(llm, "Where did it launched?")
	// Now removing embedded data and asking the same question, result should be I'm unable to provide a specific location regarding the launch of SemMapas as I don't have sufficient information on this topic.
	llm.RemoveEmbedding("")
	// Asking the same question again
	
}

func askKLLM(llm aillm.LLMContainer,query string) {
	log.Println("LLM Reply to " + query + ":")
	_, err := llm.AskLLM(query, llm.WithStreamingFunc(print))
 	if err != nil {
		log.Fatal(err)
	}
}
 
func embedd(llm aillm.LLMContainer) {
	// Text Embedding
	contents := aillm.LLMEmbeddingContent{
		Text: enRawText,
	}
	llm.EmbeddText("", contents)
}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}

const enRawText = `Welcome to SemMapas, your strategic partner in enhancing local engagement and tourism development. Designed specifically for businesses and municipalities, SemMapas offers a powerful platform to connect with residents and visitors alike, driving growth and prosperity in your community.
With SemMapas, you can effortlessly map out venues, highlight points of interest, and provide real-time updates to ensure smooth navigation for attendees. Our user-friendly interface and customizable options make it easy to tailor the experience to your specific event or business requirements.
Our platform goes beyond traditional mapping services, offering a comprehensive suite of features tailored to meet the diverse needs of event organizers and businesses alike. From tourism guides to event navigation, SemMapas empowers you to create immersive experiences that captivate your audience and enhance their journey.
Our project has been launched since 2023 in Portugal.
`


```

  

## **TODO**

  

- Implement **parallelism** to optimize processing efficiency.

- Improve **context management** to enhance memory retention and retrieval capabilities.

- Enhance the chatbot integration by supporting additional LLM models.
- Dockerized version with Reset Service and Websocket web server.

  

## **License**

  

This project is licensed under the **Apache License, Version 2.0**.
  

## **Contact**

  

For inquiries or support, reach out via:

  

- 📧 Email: reza.arani@gmail.com

- 🌐 Website: [semmapas.com](https://semmapas.com)

- 🐦 Twitter: [@RezaArani2](https://twitter.com/RezaArani2)
