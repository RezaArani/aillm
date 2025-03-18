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
		Embedder:       embeddingllmclient,
		LLMClient:      llmclient,
		ScoreThreshold: 0.6,
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}

	llm.Init()
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm)
	// Now we will ask some questions	
	askKLLM(llm, "give me a list of 3 nearby restaurants?")
	// Response should be something like: (depending on model)
	//Here are 3 nearby restaurants:
	// 1. Pizzaria & Gelataria Rialto (99 meters)
	// 2. Nicola (131 meters)
	// 3. O Consultório (150 meters)


	// asking about something within the memory
	askKLLM(llm, "What was the first choice?")
	// And result: Pizzaria & Gelataria Rialto (99 meters)

	// New we want to ask something totally different
	askKLLM(llm, "give me list of churches?")
	// based on context it must provide :
	//1. SANTA MARIA DO CASTELO CHURCH (359 meters)
	// 2. SANTO ANTÓNIO CHURCH AND CONVENT (51 meters)


	// let's go back to the first subject, Restaurants:
	askKLLM(llm, "I need a good food, what was my third option?")
	// answer: O Consultório

	// asking about distance:
	askKLLM(llm, "how far it is?")
	// answer: 150 meters


	// answer: something related to other subject 
	askKLLM(llm, "My distance to the second church?")
	// answer: 51 meters

	// Let's ask about about something that exists in the memory to check the next step.
	askKLLM(llm, "What was my last question?")

	//Let's delete User1's memory and ask the same question again. result should be something like :I'm unable to provide a specific location or time frame as I couldn't find any relevant information regarding your query.
	llm.PersistentMemoryManager.DeleteMemory("user 1")
	askKLLM(llm, "What was my last question?")

	// Cleanup
	llm.RemoveEmbedding("SemMapas")

}

func askKLLM(llm aillm.LLMContainer,   query string) {
	log.Println("LLM Reply to " + query + ":")
	queryResult, err := llm.AskLLM(query, llm.WithSessionID("user 1"), llm.WithStreamingFunc(print), llm.WithPersistentMemory(true), llm.SearchAll("en"))
	response := queryResult.Response
	resDocs := queryResult.RagDocs
	if err != nil {
		panic(err)
	}
	log.Println("CompletionTokens: ", response.Choices[0].GenerationInfo["CompletionTokens"])
	log.Println("PromptTokens: ", response.Choices[0].GenerationInfo["PromptTokens"])
	log.Println("TotalTokens: ", response.Choices[0].GenerationInfo["TotalTokens"])
	log.Println("Reference Documents: ", len(resDocs))

	for idx, doc := range resDocs {
		srcDocs := fmt.Sprintf("\t%v. Score: %v,\tSource: %s+...", idx+1, doc.Score, doc.PageContent[:50])
		log.Println(srcDocs)
	}

}

func embedd(llm aillm.LLMContainer) {
	// Text Embedding

	contents := aillm.LLMEmbeddingContent{
		Text:     Restaurant,
		Language: "en",
	}
	llm.EmbeddText("SemMapas", contents)

	contents = aillm.LLMEmbeddingContent{
		Text:     Church,
		Language: "en",
	}
	llm.EmbeddText("SemMapas", contents)

}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}

const Restaurant = `Here are some restaurants around you:

1. Pizzaria & Gelataria Rialto (99 meters)
2. Nicola (131 meters)
3. O Consultório (150 meters)
4. Restaurante Castelo (181 meters)
5. Avenida Café e Cervejaria (203 meters)
6. D. Sebastião (204 meters)
7. Biofrade (252 meters)
8. Amigos do Solar (286 meters)
`

const Church = `Here are some churches around you:

1. SANTA MARIA DO CASTELO CHURCH (359 meters)
2. SANTO ANTÓNIO CHURCH AND CONVENT (51 meters)
`
