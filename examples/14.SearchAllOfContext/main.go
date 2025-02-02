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
	embeddingIndex := "semmapas_pt"

	// let's embed some data
	log.Println("Embedding:")
	embedd(llm, embeddingIndex, "LocA",enRawText)
	embedd(llm, embeddingIndex, "LocB","Reza Arani is a golang programmer")

	askKLLMWithPrefix(llm,embeddingIndex,"LocA", "What is SemMapas?")
	askKLLMWithPrefix(llm,embeddingIndex,"LocB", "What is SemMapas?")
	askKLLMWithPrefix(llm,embeddingIndex,"LocA", "Who is Reza?")
	askKLLMWithPrefix(llm,embeddingIndex,"LocB", "Who is Reza?")

	askLLMAll(llm,embeddingIndex, "What is SemMapas?")
	askLLMAll(llm,embeddingIndex, "Who is Reza?")


	llm.RemoveEmbedding("LocA", llm.WithEmbeddingPrefix(embeddingIndex))
	llm.RemoveEmbedding("LocB", llm.WithEmbeddingPrefix(embeddingIndex))

	
}

func askKLLMWithPrefix(llm aillm.LLMContainer, prefix,index,query string) {
	log.Println("LLM Reply to " + query + ":")
	llm.AskLLM(query, llm.WithStreamingFunc(print),llm.WithEmbeddingPrefix(prefix),llm.WithEmbeddingIndex(index))
	
}

func askLLMAll(llm aillm.LLMContainer,prefix, query string) {
	log.Println("LLM Reply to " + query + ":")
	llm.AskLLM(query, llm.WithStreamingFunc(print),llm.WithEmbeddingPrefix(prefix),llm.SearchAll(""))
}


func embedd(llm aillm.LLMContainer,  prefix,indexName, contents string) {
	// Text Embedding
	LLMEmbeddingContent := aillm.LLMEmbeddingContent{
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
