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
			AiModel: "llama3.1:latest",
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

func askKLLM(llm aillm.LLMContainer, query string) {
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
	fmt.Println(string(chunk))
	return nil
}

const enRawText = `Welcome to SemMapas, your strategic partner in enhancing local engagement and tourism development. Designed specifically for businesses and municipalities, SemMapas offers a powerful platform to connect with residents and visitors alike, driving growth and prosperity in your community.
With SemMapas, you can effortlessly map out venues, highlight points of interest, and provide real-time updates to ensure smooth navigation for attendees. Our user-friendly interface and customizable options make it easy to tailor the experience to your specific event or business requirements.
Our platform goes beyond traditional mapping services, offering a comprehensive suite of features tailored to meet the diverse needs of event organizers and businesses alike. From tourism guides to event navigation, SemMapas empowers you to create immersive experiences that captivate your audience and enhance their journey.
Our project has been launched since 2023 in Portugal.
`
