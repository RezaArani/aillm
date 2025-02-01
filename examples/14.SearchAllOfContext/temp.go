package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	aillm "github.com/RezaArani/aillm/controller"
)

func main() {
	log.Println("Start:")
	log.Println(time.Now())
	// locally hosted ollama

	embeddingllmclient := &aillm.OllamaController{
		Config: aillm.LLMConfig{
			Apiurl:  "http://127.0.0.1:11434",
			AiModel: "all-minilm",
		},
	}

	llmclient := &aillm.OpenAIController{
		Config: aillm.LLMConfig{
			Apiurl:   "https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			AiModel:  "Meta-Llama-3_1-70B-Instruct",
			APIToken: os.Getenv("APITOKEN"),
		},
	}

	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		Embedder:  embeddingllmclient,
		LLMClient: llmclient,
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}

	llm.Init()
	embeddingIndex := "rawText"

	// let's embed some data
	log.Println("Embedding:")
	embedd(llm, embeddingIndex, "A",enRawText)
	embedd(llm, embeddingIndex, "B","Reza Arani is a golang programmer")

	askKLLMWithPrefix(llm,"A", "What is SemMapas?")
	askKLLMWithPrefix(llm,"B", "What is SemMapas?")
	askKLLMWithPrefix(llm,"A", "Who is Reza?")
	askKLLMWithPrefix(llm,"B", "Who is Reza?")

	askLLMAll(llm, "What is SemMapas?")
	askLLMAll(llm, "Who is Reza?")


	llm.RemoveEmbedding(embeddingIndex, llm.WithEmbeddingPrefix("A"))
	llm.RemoveEmbedding(embeddingIndex, llm.WithEmbeddingPrefix("B"))

	
}

func askKLLMWithPrefix(llm aillm.LLMContainer, prefix,query string) {
	log.Println("LLM Reply to " + query + ":")
	llm.AskLLM(query, llm.WithStreamingFunc(print),llm.WithEmbeddingPrefix(prefix))
	
}

func askLLMAll(llm aillm.LLMContainer, query string) {
	log.Println("LLM Reply to " + query + ":")
	llm.AskLLM(query, llm.WithStreamingFunc(print),llm.SearchAll("en"))
}


func embedd(llm aillm.LLMContainer, indexName, prefix, contents string) {
	// Text Embedding
	LLMEmbeddingContent := make(map[string]aillm.LLMEmbeddingContent)
	LLMEmbeddingContent["en"] = aillm.LLMEmbeddingContent{
		Text: contents,
	}

	llm.EmbeddText(indexName, LLMEmbeddingContent, llm.WithEmbeddingPrefix(prefix))

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
