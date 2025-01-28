package main

import (
	"context"
	"fmt"
	"log"

	aillm "github.com/RezaArani/aillm/controller"
)

func main() {
	//define client

	
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

	//index name
	embeddingTitle := "SemMapas Contents"

	// Text Embedding
	contents := make(map[string]aillm.LLMEmbeddingContent)
	contents["en"] = aillm.LLMEmbeddingContent{
		Text: `Welcome to SemMapas, your strategic partner in enhancing local engagement and tourism development. Designed specifically for businesses and municipalities, SemMapas offers a powerful platform to connect with residents and visitors alike, driving growth and prosperity in your community.
	With SemMapas, you can effortlessly map out venues, highlight points of interest, and provide real-time updates to ensure smooth navigation for attendees. Our user-friendly interface and customizable options make it easy to tailor the experience to your specific event or business requirements.
	Our platform goes beyond traditional mapping services, offering a comprehensive suite of features tailored to meet the diverse needs of event organizers and businesses alike. From tourism guides to event navigation, SemMapas empowers you to create immersive experiences that captivate your audience and enhance their journey.
	Our project has been launched since 2023 in Portugal.
	`,
	}

	llm.EmbeddText(embeddingTitle, contents)
	// time to call LLM, Now it knows what is SemMapas
	queryResult, err := llm.AskLLM("What is SemMapas?", llm.WithStreamingFunc(print))
	if err != nil {
		log.Fatal(err)
	}

	llm.RemoveEmbeddingDataFromRedis(embeddingTitle)
	log.Println("TotalTokens: ", queryResult.Response.Choices[0].GenerationInfo["TotalTokens"])

}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
