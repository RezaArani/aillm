package main

import (
	"context"
	"fmt"
	"log"
	"os"

	aillm "github.com/RezaArani/aillm/controller"
)

func main() {
	log.Println("Start:")

	// This test utilizes different models and clients to demonstrate functionality:
	// - For embeddings, we use the "all-MiniLM" model.
	// - For the LLM component, we leverage OVHCloud AI endpoints via the OpenAI client.
	// Additionally, the OLLAMA model is hosted locally for testing purposes, showcasing the integration of various systems.

	embeddingllmclient := &aillm.OllamaController{
		Config: aillm.LLMConfig{
			Apiurl:  "http://127.0.0.1:11434",
			AiModel: "mxbai-embed-large",
		},
	}

	llmclient := &aillm.OpenAIController{
		Config: aillm.LLMConfig{
			Apiurl:   "https://llama-3-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			AiModel:  "Meta-Llama-3_3-70B-Instruct",
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
		ScoreThreshold: 0.1,
	}
	llm.Init()
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm)
	// Distance to the result. greater means more R data but not accurate
	queryResult, err := llm.AskLLM("SemMapas city and clients?", llm.WithStreamingFunc(print))
	
	if err != nil {
		panic(err)
	}
	refrences, err := llm.GetRagIndexs(queryResult.RagDocs)
	
	if err != nil {
		panic(err)
	}
	fmt.Println("\nRefrences:")
	for _idx, refrence := range refrences {
		fmt.Println("\t",_idx+1, refrence)
	}

	// Cleanup
	llm.RemoveEmbedding("MyData")

}
 
func embedd(llm aillm.LLMContainer) {
	// Text Embedding
	contents := aillm.LLMEmbeddingContent{
		Text: SemMapas,
	}
	_, err := llm.EmbeddText("MyData", contents)
	if err != nil {
		panic(err)
	}
}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}

const SemMapas = `Welcome to SemMapas, your strategic partner in enhancing local engagement and tourism development. Designed specifically for businesses and municipalities, SemMapas offers a powerful platform to connect with residents and visitors alike, driving growth and prosperity in your community.
With SemMapas, you can effortlessly map out venues, highlight points of interest, and provide real-time updates to ensure smooth navigation for attendees. Our user-friendly interface and customizable options make it easy to tailor the experience to your specific event or business requirements.
Our platform goes beyond traditional mapping services, offering a comprehensive suite of features tailored to meet the diverse needs of event organizers and businesses alike. From tourism guides to event navigation, SemMapas empowers you to create immersive experiences that captivate your audience and enhance their journey.
Our project has been launched since 2023 in Portugal and city of Lourinhã.
`
