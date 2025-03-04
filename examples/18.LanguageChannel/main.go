package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	aillm "github.com/RezaArani/aillm/controller"
)

func main() {
	log.Println("Start:")

	llmclient := &aillm.OpenAIController{
		Config: aillm.LLMConfig{
			Apiurl:   "https://llama-3-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			AiModel:  "Meta-Llama-3_3-70B-Instruct",
			APIToken: os.Getenv("APITOKEN"),
			// APIToken:  os.Getenv("OPENAIAPITOKEN"),
			// Apiurl:  "https://api.openai.com/v1/",
			// AiModel: "gpt-4o",
		},
	}

	//

	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		Embedder:       llmclient,
		LLMClient:      llmclient,
		ScoreThreshold: 0.6,
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
		AllowHallucinate:                    true,
		LLMModelLanguageDetectionCapability: true,
	}

	llm.Init()
	// let's embed some data

	// Now we will ask some questions
	ch := make(chan string)
	var w sync.WaitGroup

	go func() {
		w.Add(1)
		_, err := llm.AskLLM(
			"سلام. چطوری؟",
			llm.WithLanguageChannel(ch),
			llm.WithStreamingFunc(print),
		)
		if err != nil {
			panic(err)
		}
		close(ch)

		w.Done()
	}()

	select {
	case msg := <-ch:
		fmt.Println("\nLanguage:", msg)
	case <-time.After(1 * time.Second):
		fmt.Println("Timeout!")
	}
	w.Wait()
}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
