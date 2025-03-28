package main

import (
	"context"
	"fmt"
	"log"

	aillm "github.com/RezaArani/aillm/controller"
)

func main() {
	log.Println("Start:")

	llmclient := &aillm.OllamaController{
		Config: aillm.LLMConfig{
			Apiurl:  "http://127.0.0.1:11434",
			AiModel: "llama3.1",
		},
	}
 

	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		LLMClient: llmclient,
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	

	llm.Init()
	embeddingIndex := "rawText"

	// let's embed some data
	log.Println("Embedding:")
	embedd(llm, embeddingIndex)
	// looks for the data in "en" language contents
	askKLLM(llm, embeddingIndex, "en", "What is SemMapas?")
	// looks for the data in "pt" language contents but replies in English
	askKLLM(llm, embeddingIndex, "pt", "What is SemMapas?")
	// looks for the data in "en" language contents, so based on enRawText sample, result should be Portugal
	askKLLM(llm, embeddingIndex, "en", "Where did it launched?")
	// looks for the data in "pt" language contents, so based on sample, result should be Lourinhã
	askKLLM(llm, embeddingIndex, "pt", "Where did it launched?")
	// looks for the data in "pt" language contents, so based on ptRawText sample, result should be April 2023
	askKLLM(llm, embeddingIndex, "pt", "When did it launched?")
	// looks for the data in "pt" language contents, so based on ptRawText sample, result should be just 2023
	askKLLM(llm, embeddingIndex, "en", "When did it launched?")
	// Now removing embedded data and asking the same question, result should be I'm unable to provide a specific location regarding the launch of SemMapas as I don't have sufficient information on this topic.
	llm.RemoveEmbedding(embeddingIndex)
	askKLLM(llm,embeddingIndex, "pt", "Where did it launched?")

}

func askKLLM(llm aillm.LLMContainer, index, Language, query string) {
	log.Println("LLM Reply to " + query + ":")
	queryResult, err := llm.AskLLM(query, llm.WithLanguage(Language), llm.WithStreamingFunc(print))
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

func embedd(llm aillm.LLMContainer, Index string) {
	// Text Embedding
	
	contents:= aillm.LLMEmbeddingContent{
		Text: enRawText,
		Language: "en",
	}
	llm.EmbeddText(Index, contents)


	contentsPt:= aillm.LLMEmbeddingContent{
		Text: ptRawText,
		Language: "pt",
	}
	 
	llm.EmbeddText(Index, contentsPt)

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
const ptRawText = `Bem-vindo ao SemMapas, o seu parceiro estratégico na melhoria do envolvimento local e no desenvolvimento do turismo. Projetado especificamente para empresas e municípios, o SemMapas oferece uma plataforma poderosa para conectar-se com residentes e visitantes, impulsionando o crescimento e a prosperidade na sua comunidade.
Com o SemMapas, pode facilmente mapear locais, destacar pontos de interesse e fornecer atualizações em tempo real para garantir uma navegação suave para os participantes. A nossa interface fácil de usar e as opções personalizáveis tornam simples adaptar a experiência às necessidades específicas do seu evento ou negócio.
A nossa plataforma vai além dos serviços de mapeamento tradicionais, oferecendo um conjunto abrangente de funcionalidades ajustadas para responder às diversas necessidades de organizadores de eventos e empresas. Desde guias turísticos até à navegação em eventos, o SemMapas capacita-o a criar experiências imersivas que cativam o seu público e enriquecem a sua jornada.
O nosso projeto está lançado desde Abril 2023 em Lourinã.`
