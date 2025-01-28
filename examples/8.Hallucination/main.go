package main

import (
	"context"
	"fmt"
	"log"
	"os"

	aillm "github.com/RezaArani/aillm/controller"
	"github.com/tmc/langchaingo/schema"
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
		Embedder:  embeddingllmclient,
		LLMClient: llmclient,
		DataRedis: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	llm.Init()
	embeddingId := "MyIdDifferentModels"
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm, embeddingId)
	// Rag powered query 
	askKLLM(llm, embeddingId, "en", "User1", "What is SemMapas?")
	// Now let's remove the embedding and the result should be something like I couldn't find any relevant information or a clear answer regarding your question about SemMapas.
	log.Println("Removing Embedding:")
	llm.RemoveEmbeddingDataFromRedis(embeddingId, "SemMapas")
	askKLLM(llm, embeddingId, "en", "User2", "What is SemMapas?")
	// Now let's rely on model data and hallucination and the result should be something like "SemMapas is a Brazilian navigation app that provides turn-by-turn directions and real-time traffic information." which is not correct.
	llm.AllowHallucinate = true
	askKLLM(llm, embeddingId, "en", "User3", "What is SemMapas?")

	// Cleanup
	removeEmbedd(llm, embeddingId, "SemMapas")
	

}

func askKLLM(llm aillm.LLMContainer, EmbeddingId, Language, user, query string) {
	log.Println("LLM Reply to " + query + " from " + user + ":")
	response, resDocs, err := llm.AskLLM(EmbeddingId, Language, user, query, print)
	if err!=nil{
		panic(err)
	}
	log.Println("CompletionTokens: ", response.Choices[0].GenerationInfo["CompletionTokens"])
	log.Println("PromptTokens: ", response.Choices[0].GenerationInfo["PromptTokens"])
	log.Println("TotalTokens: ", response.Choices[0].GenerationInfo["TotalTokens"])
	log.Println("Reference Documents: ", len(resDocs.([]schema.Document)))

	for idx, doc := range resDocs.([]schema.Document) {
		srcDocs := fmt.Sprintf("\t%v. Score: %v,\tSource: %s+...", idx+1, doc.Score, doc.PageContent[:50])
		log.Println(srcDocs)
	}
	 
}

func removeEmbedd(llm aillm.LLMContainer, embeddingId, title string) {
	llm.RemoveEmbeddingDataFromRedis(embeddingId, title)
}
func embedd(llm aillm.LLMContainer, EmbeddingId string) {
	// Text Embedding
	contents := make(map[string]aillm.LLMEmbeddingContent)
	contents["en"] = aillm.LLMEmbeddingContent{
		Text: SemMapas,
	}
	llm.EmbeddText(EmbeddingId, "SemMapas", contents)
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
