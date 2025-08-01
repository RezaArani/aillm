// Copyright (c) 2025 Reza Arani
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package aillm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/tmc/langchaingo/schema"
)

// PersistentMemory structure to store user memory session data in a persistent storage (Redis) for future retrival or vector search.
//
// This struct keeps track of a user's questions parameters.
//
// Fields:
//   - Questions: A slice of strings representing the list of user queries in the session.
//   - MemoryStartTime: A timestamp indicating when the session started.
type PersistentMemory struct {
	MemoryPrefix          string        //prefix for redis storage
	MemoryTTL             time.Duration // auto delete memory question TTL
	MemorySearchThreshold float32       //Memory vector search Threshold
	HistoryItemCount      int           // More queries = more tokens. adjus it carefully.
	redisClient           *redis.Client // Redis client for persistent storage
	lLMContainer          *LLMContainer // LLM container for embedding and vector search
}

// initPersistentMemoryManager initializes the persistent memory manager based on default configuration.
//
// Returns:
//   - error: Returns an error if initialization fails or if the provider is unsupported.
func (llm *LLMContainer) initPersistentMemoryManager() {

	persistentMemory := &PersistentMemory{
		redisClient:           llm.RedisClient.redisClient,
		MemoryPrefix:          "Memory",
		MemoryTTL:             30 * time.Minute,
		lLMContainer:          llm,
		MemorySearchThreshold: llm.ScoreThreshold,
		HistoryItemCount:      1,
	}
	llm.PersistentMemoryManager = *persistentMemory

}

// AddMemory stores user questions in Redis and Embeds query to vector database for future related questions.
//
// Parameters:
//   - sessionID: The unique identifier for the user's session.
//   - query: A pair of Query and Answer and Keys slice (Saved in Redis).

// Returns:
//   - error: An error if the embedding process fails.
func (pm *PersistentMemory) AddMemory(sessionID string, query MemoryData) (TokenUsage, error) {
	tokenUsage := TokenUsage{}
	embeddingPrefix := pm.MemoryPrefix + ":" + sessionID + ":aillm_vector_idx"

	promotPart := fmt.Sprintf("\nUser: %v\nAssistant: %v\n\n", query.Question, query.Answer)
	memoryembeddingContent := LLMEmbeddingContent{
		Title: promotPart,
	}

	keys, _, _, _, err := pm.lLMContainer.embedText("Memory", "aillm", embeddingPrefix, "", promotPart, "", memoryembeddingContent, true, true, false)
	//
	//Updating redis TTL

	for _, key := range keys {
		pm.redisClient.Expire(context.TODO(), key, pm.MemoryTTL)
	}
	if err != nil {
		return tokenUsage, err
	}
	query.Keys = keys
	// fetch previous memory from Redis
	curUserMemoryStr := pm.redisClient.Get(context.TODO(), "rawMemory:"+pm.MemoryPrefix+":"+sessionID).Val()
	curUserMemory := Memory{}

	if curUserMemoryStr != "" {
		err = json.Unmarshal([]byte(curUserMemoryStr), &curUserMemory)
		if err != nil {
			return tokenUsage, err
		}
	}

	curUserMemory.Questions = append(curUserMemory.Questions, query)

	if len(curUserMemory.Questions) >= 2 {
		PrevConversation := ""
		for _, question := range curUserMemory.Questions {
			if question.Answer[0] == '@' {
				question.Answer = question.Answer[1:]
			}
			PrevConversation += fmt.Sprintf("User: %v\nAssistant: %v\n\n", question.Question, question.Answer)
		}
		resp, err := pm.lLMContainer.AskLLM("", pm.lLMContainer.WithExactPrompt("You are a helpful assistant that summarizes conversations as short as possible with details for future use of LLM memory.\n"+PrevConversation), pm.lLMContainer.WithAllowHallucinate(true), pm.lLMContainer.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			tokenUsage.OutputTokens++
			return nil
		}))
		if err != nil {
			return tokenUsage, err
		}
		curUserMemory.Summary = resp.Response.Choices[0].Content
	}

	curUserMemoryBytes, err := json.Marshal(curUserMemory)
	if err != nil {
		return tokenUsage, err
	}
	err = pm.redisClient.Set(context.TODO(), "rawMemory:"+pm.MemoryPrefix+":"+sessionID, string(curUserMemoryBytes), pm.MemoryTTL).Err()

	return tokenUsage, err
}

// GetMemory retrieves stored session memory for a given session ID.
//
// The function safely reads from the session map and returns the stored memory if it exists.
//
// Parameters:
//   - sessionID: The unique identifier for the user's session.
//   - query: Will be used for Vector search in user query history to find previous related questions
//
// Returns:
//   - MemoryData: Last asked question.
//   - string: generated prompt for memory context.
//   - error: An error if the memory retrival process fails.
func (pm *PersistentMemory) GetMemory(sessionID string, query string) (MemoryData, Memory, string, []schema.Document, error) {
	result := ""
	curUserMemory := Memory{}

	memoryhistory := []schema.Document{}

	// Get last question from Memory

	_, err := pm.redisClient.Ping(context.TODO()).Result()
	if err != nil {
		return MemoryData{}, curUserMemory, "", memoryhistory, err
	}

	redisCmd := pm.redisClient.Get(context.TODO(), "rawMemory:"+pm.MemoryPrefix+":"+sessionID)
	lastQuestion := MemoryData{}
	if redisCmd.Err() != nil {
		return lastQuestion, curUserMemory, "", memoryhistory, redisCmd.Err()
	}
	curUserMemoryStr := redisCmd.Val()
	_ = json.Unmarshal([]byte(curUserMemoryStr), &curUserMemory)
	if curUserMemory.Summary != "" {
		result = "Memory Summary: " + curUserMemory.Summary + "\n"
	}
	if len(curUserMemory.Questions) > 0 {
		embeddingPrefix := "Memory:" + pm.MemoryPrefix + ":" + sessionID + ":"
		lastQuestion = curUserMemory.Questions[len(curUserMemory.Questions)-1]
		if len(curUserMemory.Questions) > 1 {
			// secondLastQuestion := curUserMemory.Questions[len(curUserMemory.Questions)-2]
			// result += "User: " + secondLastQuestion.Question + "\nAssistant:" + secondLastQuestion.Answer + "\n"
			resDocs, searchErr := pm.lLMContainer.CosineSimilarity(embeddingPrefix, query, pm.HistoryItemCount, pm.MemorySearchThreshold)
			err = searchErr

			for _, doc := range resDocs {
				result += doc.PageContent
				memoryhistory = append(memoryhistory, doc)
			}

		}

		result += "User: " + lastQuestion.Question + "\nAssistant:" + lastQuestion.Answer + "\n"
	}
	return lastQuestion, curUserMemory, result, memoryhistory, err
}

// DeleteMemory removes a user's session memory from the memory map.
//
// Parameters:
//   - sessionID: The unique identifier for the session to be deleted.
func (pm *PersistentMemory) DeleteMemory(sessionID string) error {
	// llm.userLanguage[o.SessionID]
	if sessionID == "" {
		return nil
	}
	if pm.lLMContainer.userLanguage != nil {
		pm.lLMContainer.userLanguage[sessionID] = ""
	}
	keyPrefix := "rawMemory:" + pm.MemoryPrefix + ":" + sessionID
	redisCmd := pm.redisClient.Get(context.TODO(), keyPrefix)
	redisCmdErr := redisCmd.Err()
	if redisCmdErr != nil {
		return redisCmdErr
	}
	curUserMemoryStr := redisCmd.Val()
	curUserMemory := Memory{}
	_ = json.Unmarshal([]byte(curUserMemoryStr), &curUserMemory)
	for _, mem := range curUserMemory.Questions {
		for _, key := range mem.Keys {
			pm.redisClient.Del(context.TODO(), key)
		}
	}
	var err error
	rawMemErr := pm.redisClient.Del(context.TODO(), keyPrefix).Err()
	if rawMemErr != nil {
		err = errors.New(rawMemErr.Error())
	}
	return err
}
