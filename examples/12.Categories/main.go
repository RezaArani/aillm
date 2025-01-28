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
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	llm.Init()
	// let's embed some data
	log.Println("Embedding:")
	embedd(llm)
	// First search will be in "Company category". Result should be something about SemMapas company.
	askKLLM(llm, "Company", "SemMapas city?")
	// Now let's search in Agriculture category and result should be something like I can't find ...
	askKLLM(llm, "Agriculture", "SemMapas city?")

	// Now let's search in Agriculture category about related subject.
	askKLLM(llm, "Agriculture", "Tell me about crop infection.")

	// Cleanup
	llm.RemoveEmbeddingDataFromRedis("CropPestInfection", llm.WithEmbeddingPrefix("Agriculture"))
	llm.RemoveEmbeddingDataFromRedis("SemMapas", llm.WithEmbeddingPrefix("Company"))

}

func askKLLM(llm aillm.LLMContainer, category, query string) {
	log.Println("LLM Reply to " + ":")
	queryResult, err := llm.AskLLM(query, llm.WithStreamingFunc(print), llm.WithEmbeddingPrefix(category))
	response := queryResult.Response
	resDocs := queryResult.RagDocs
	if err != nil {
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

func embedd(llm aillm.LLMContainer) {
	// Text Embedding
	contents := make(map[string]aillm.LLMEmbeddingContent)
	contents["en"] = aillm.LLMEmbeddingContent{
		Text: SemMapas,
	}
	//llm.WithEmbeddingPrefix() will provide searching in a different category
	llm.EmbeddText("SemMapas", contents, llm.WithEmbeddingPrefix("Company"))

	contents = make(map[string]aillm.LLMEmbeddingContent)
	contents["en"] = aillm.LLMEmbeddingContent{
		Text: CropPestInfection,
	}
	llm.EmbeddText("CropPestInfection", contents, llm.WithEmbeddingPrefix("Agriculture"))

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

const CropPestInfection = `Crop pest infections pose a significant threat to global agriculture, affecting both food security and economic stability. Pests such as insects, fungi, nematodes, and weeds can devastate crops by feeding on plants, spreading diseases, or competing for nutrients. Common pests like locusts, aphids, and armyworms can quickly infest fields, leading to severe yield losses, particularly in regions heavily reliant on agriculture. For example, locust swarms have been known to destroy thousands of hectares of crops within days, causing famine and financial hardship. Factors such as climate change, monoculture farming, and the misuse of pesticides have exacerbated pest outbreaks, creating challenges for farmers worldwide.
Addressing crop pest infections requires an integrated pest management (IPM) approach, combining cultural, biological, and chemical methods to minimize damage. Practices such as crop rotation, resistant crop varieties, and natural predators like ladybugs and parasitic wasps help control pest populations sustainably. Innovations in technology, including drone surveillance, precision agriculture, and biopesticides, have further enhanced farmers' ability to detect and combat pests early. However, global collaboration, research, and education are essential to develop scalable solutions, particularly for smallholder farmers in vulnerable regions. By investing in these strategies, it is possible to mitigate the impact of pest infections and ensure a stable food supply for growing populations. There are a lot of farms exists nearby Paris.`
