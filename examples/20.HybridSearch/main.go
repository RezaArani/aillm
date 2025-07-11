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
	log.Println("=== Hybrid Search Demonstration ===")

	// Setup embedding and LLM clients
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

	// Create LLM instance
	llm := aillm.LLMContainer{
		Embedder:       embeddingllmclient,
		LLMClient:      llmclient,
		ScoreThreshold: 0.3, // Lower threshold for better recall
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
	}
	llm.Init()

	// Embed sample data
	log.Println("📚 Embedding sample data...")
	embedSampleData(llm)

	// Test different search algorithms
	log.Println("\n🔍 Testing different search algorithms...")

	// Test query that benefits from hybrid search
	query := "artificial intelligence machine learning"

	// 1. Traditional Vector Search
	log.Println("\n1️⃣ Traditional Vector Search (Cosine Similarity):")
	testSearchAlgorithm(llm, query, llm.WithSimilaritySearch())

	// 2. KNN Search
	log.Println("\n2️⃣ K-Nearest Neighbors Search:")
	testSearchAlgorithm(llm, query, llm.WithKNNSearch())

	// 3. Lexical Search Only
	log.Println("\n3️⃣ Lexical Search Only:")
	testSearchAlgorithm(llm, query, llm.WithLexicalSearch())

	// 4. Hybrid Search (Vector + Lexical)
	log.Println("\n4️⃣ Hybrid Search (Vector + Lexical):")
	testSearchAlgorithm(llm, query, llm.WithHybridSearch())

	// 5. Semantic Search (Auto-selects best algorithm)
	log.Println("\n5️⃣ Semantic Search (Auto-optimized):")
	testSearchAlgorithm(llm, query, llm.WithSemanticSearch())

	// Test with different types of queries
	log.Println("\n🧪 Testing with different query types...")

	// Test exact keyword match
	log.Println("\n📝 Exact Keyword Match Test:")
	testQueryComparison(llm, "Python programming")

	// Test semantic similarity
	log.Println("\n🧠 Semantic Similarity Test:")
	testQueryComparison(llm, "What is AI?")

	// Test mixed query
	log.Println("\n🔄 Mixed Query Test:")
	testQueryComparison(llm, "deep learning neural networks")

	// Cleanup
	log.Println("\n🧹 Cleaning up...")
	llm.RemoveEmbedding("HybridSearchDemo")
	log.Println("✅ Demo completed!")
}

func embedSampleData(llm aillm.LLMContainer) {
	sampleData := []aillm.LLMEmbeddingContent{
		{
			Id:       "ai-basics",
			Title:    "Introduction to Artificial Intelligence",
			Text:     `Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.`,
			Keywords: []string{"artificial intelligence", "AI", "machine learning", "deep learning", "neural networks"},
			Sources:  "AI_Textbook_Chapter1.pdf",
		},
		{
			Id:       "python-ml",
			Title:    "Python for Machine Learning",
			Text:     `Python is the most popular programming language for machine learning and artificial intelligence development. It offers powerful libraries like scikit-learn, TensorFlow, and PyTorch that make it easy to build and train machine learning models. Python's simple syntax and extensive ecosystem make it ideal for data science, statistical analysis, and AI research.`,
			Keywords: []string{"Python", "programming", "machine learning", "scikit-learn", "TensorFlow", "PyTorch"},
			Sources:  "Python_ML_Guide.pdf",
		},
		{
			Id:       "deep-learning",
			Title:    "Deep Learning and Neural Networks",
			Text:     `Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition, natural language processing, and speech recognition. Popular frameworks include TensorFlow, PyTorch, and Keras.`,
			Keywords: []string{"deep learning", "neural networks", "TensorFlow", "PyTorch", "Keras", "image recognition"},
			Sources:  "Deep_Learning_Fundamentals.pdf",
		},
		{
			Id:       "data-science",
			Title:    "Data Science and Analytics",
			Text:     `Data science combines statistics, programming, and domain expertise to extract meaningful insights from data. It involves collecting, cleaning, analyzing, and interpreting large datasets to support decision-making. Data scientists use tools like Python, R, SQL, and various visualization libraries to uncover patterns and trends in data.`,
			Keywords: []string{"data science", "analytics", "statistics", "Python", "R", "SQL", "visualization"},
			Sources:  "Data_Science_Handbook.pdf",
		},
		{
			Id:       "blockchain",
			Title:    "Blockchain Technology",
			Text:     `Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. Blockchain technology is the foundation of cryptocurrencies like Bitcoin and Ethereum.`,
			Keywords: []string{"blockchain", "distributed ledger", "cryptocurrency", "Bitcoin", "Ethereum", "cryptography"},
			Sources:  "Blockchain_Basics.pdf",
		},
	}

	for _, content := range sampleData {
		_, err := llm.EmbeddText("HybridSearchDemo", content)
		if err != nil {
			log.Printf("Error embedding content %s: %v", content.Id, err)
		}
	}
}

func testSearchAlgorithm(llm aillm.LLMContainer, query string, searchOption aillm.LLMCallOption) {
	result, err := llm.AskLLM(query, searchOption, llm.WithStreamingFunc(print))
	if err != nil {
		log.Printf("❌ Error: %v", err)
		return
	}

	log.Printf("📊 Found %d relevant documents", len(result.RagDocs))
	for i, doc := range result.RagDocs {
		log.Printf("   %d. Score: %.3f - %s", i+1, doc.Score, getDocumentTitle(doc))

		// Show search type if available
		if searchType, ok := doc.Metadata["search_type"]; ok {
			log.Printf("      Search Type: %s", searchType)
		}

		// Show hybrid scores if available
		if hybridScore, ok := doc.Metadata["hybrid_score"]; ok {
			log.Printf("      Hybrid Score: %.3f", hybridScore)
		}
		if vectorScore, ok := doc.Metadata["vector_score"]; ok {
			log.Printf("      Vector Score: %.3f", vectorScore)
		}
		if lexicalScore, ok := doc.Metadata["lexical_score"]; ok {
			log.Printf("      Lexical Score: %.3f", lexicalScore)
		}
	}
}

func testQueryComparison(llm aillm.LLMContainer, query string) {
	log.Printf("Query: '%s'", query)

	// Test with different search algorithms
	algorithms := []struct {
		name   string
		option aillm.LLMCallOption
	}{
		{"Vector", llm.WithSimilaritySearch()},
		{"Lexical", llm.WithLexicalSearch()},
		{"Hybrid", llm.WithHybridSearch()},
	}

	for _, alg := range algorithms {
		result, err := llm.AskLLM(query, alg.option)
		if err != nil {
			log.Printf("   %s: ❌ Error: %v", alg.name, err)
			continue
		}

		log.Printf("   %s: Found %d docs", alg.name, len(result.RagDocs))
		if len(result.RagDocs) > 0 {
			log.Printf("        Top result: %s (Score: %.3f)",
				getDocumentTitle(result.RagDocs[0]), result.RagDocs[0].Score)
		}
	}
}

func getDocumentTitle(doc schema.Document) string {
	if title, ok := doc.Metadata["title"]; ok {
		return fmt.Sprintf("%v", title)
	}
	if len(doc.PageContent) > 50 {
		return doc.PageContent[:50] + "..."
	}
	return doc.PageContent
}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
