// Copyright (c) 2025 John Doe
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
	"fmt"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// LLMEmbeddingContent represents a single piece of text content that is embedded and stored in the system.
//
// This struct holds the necessary information for managing and identifying embedded text,
// including the raw text, a title, its source, and associated keys.
//
// Fields:
//   - Text: The raw text content that is embedded.
//   - Title: A descriptive title for the embedded content.
//   - Index:
//   - Source: The origin of the content, such as a file name, URL, or other identifier.
//   - Keys: A slice of strings representing the Redis keys associated with this content.
type LLMEmbeddingContent struct {
	Text        string `json:"Text" redis:"Text"`
	Title       string `json:"Title" redis:"Title"`
	Language    string `json:"Language" redis:"Language"`
	Id          string `json:"Id" redis:"Id"`
	Source      string `json:"Source" redis:"Source"`
	Keys        []string
	GeneralKeys []string
}

// LLMEmbeddingObject represents a collection of embedded text contents grouped under a specific object ID.
//
// This struct serves as a container for multiple pieces of embedded text content, organized by language or context.
// It provides a way to store and manage embeddings for a specific use case or document.
//
// Fields:
//   - EmbeddingPrefix: A unique prefix or identifier for the embedding object (e.g., "ObjectId").
//   - Index: An Index for the embedding object, providing a future access.
//   - Contents: A map of language-specific content, where the key is the language code (e.g., "en", "pt")
//     and the value is an LLMEmbeddingContent struct containing the associated content details.
type LLMEmbeddingObject struct {
	EmbeddingPrefix string                         `json:"EmbeddingPrefix" redis:"EmbeddingPrefix"`
	Index           string                         `json:"Index" redis:"Index"`
	Contents        map[string]LLMEmbeddingContent `json:"Contents" redis:"Contents"`
}

// getRawDocRedisId generates a unique Redis key for storing raw document data.
// It combines the object ID and a sanitized version of the Index to create a consistent key format.
//
// Returns:
//   - A string representing the Redis key in the format "rawDocs:ObjectId:Index".
func (llmeo LLMEmbeddingObject) getRawDocRedisId() string {
	// Construct Redis key using object ID and sanitized Index
	return "rawDocs:" + llmeo.EmbeddingPrefix + ":" + llmeo.sanitizeRedisKey(llmeo.Index)
}

// EmbeddFile processes and embeds the content of a given file into the LLM system.
//
// Parameters:
//   - ObjectId: A unique identifier for the embedding object.
//   - Title: The Title of the document being embedded. Also it will be used for raw data for a better Context
//   - Index: The Index of the document being embedded. Also it will be used for the raw file redis key
//   - fileName: The path to the file to be embedded.
//   - tc: Configuration for transcription, such as language settings.
//
// Returns:
//   - LLMEmbeddingObject: The embedded object containing the processed content.
//   - error: An error if any issues occur during processing.
func (llm LLMContainer) EmbeddFile(Index, Title, fileName string, tc TranscribeConfig, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	var result LLMEmbeddingObject
	// EmbeddingContents := make(map[string]LLMEmbeddingContent)
	// Transcribe the file to extract text content
	fileContents, _, transcribeErr := llm.Transcriber.transcribeFile(fileName, "", tc)
	if transcribeErr != nil {
		return result, transcribeErr
	}

	// Store transcribed content with language as key
	EmbeddingContents := LLMEmbeddingContent{
		Text:   fileContents,
		Title:  Title,
		Source: fileName,
	}

	// Embed the transcribed text into the LLM system
	embeddedTextObjects, embedErr := llm.EmbeddText(Index, EmbeddingContents, options...)
	if embedErr != nil {
		return result, embedErr
	}
	return embeddedTextObjects, embedErr
}

// EmbeddURL processes and embeds content from a given URL into the LLM system.
//
// Parameters:
//   - ObjectId: A unique identifier for the embedding object.
//   - Index: The Index associated with the content being embedded.
//   - url: The web URL from which content will be transcribed and embedded.
//   - tc: Configuration for transcription, including language and extraction options.
//
// Returns:
//   - LLMEmbeddingObject: The embedded object containing the processed content.
//   - error: An error if any issues occur during the transcription or embedding process.
func (llm LLMContainer) EmbeddURL(Index, url string, tc TranscribeConfig, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	var result LLMEmbeddingObject
	// Transcribe the content from the provided URL
	fileContents, _, transcribeErr := llm.Transcriber.transcribeURL(url, tc)
	if transcribeErr != nil {
		return result, transcribeErr
	}

	// Store transcribed content with the specified language as key
	EmbeddingContents := LLMEmbeddingContent{
		Text:   fileContents,
		Source: url,
	}

	// Embed the transcribed text into the LLM system
	// o := LLMCallOptions{}
	// for _, opt := range options {opt(&o)}
	embeddedTextObjects, embedErr := llm.EmbeddText(Index, EmbeddingContents, options...)
	if embedErr != nil {
		return result, embedErr
	}
	// redisErr := llm.saveEmbeddingDataToRedis(embeddedTextObjects)
	return embeddedTextObjects, embedErr

}

// EmbeddText embeds provided text content into the LLM system and stores the data in Redis.
//
// Parameters:
//   - ObjectId: Unique identifier for the embedding object.
//   - Index: Index associated with the content being embedded.
//   - Contents: A map containing language-specific content to be embedded.
//
// Returns:
//   - LLMEmbeddingObject: The resulting embedding object after processing and storage.
//   - error: An error if any issues occur during embedding or Redis operations.
func (llm *LLMContainer) EmbeddText(Index string, Contents LLMEmbeddingContent, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}

	result := LLMEmbeddingObject{
		EmbeddingPrefix: o.getEmbeddingPrefix(),
		Index:           Index,
	}

	// Load existing data from Redis if available
	result.load(llm.RedisClient.redisClient, result.getRawDocRedisId())

	// Process embedding for each language in contents
	if result.Contents == nil {
		result.Contents = make(map[string]LLMEmbeddingContent)
	}
	if Contents.Id == "" {
		Contents.Id = uuid.New().String()
	}
	//
	tempKeys, generalKeys, _, err := llm.embedText(o.getEmbeddingPrefix(), Contents.Language, Index, Contents.Title, llm.Transcriber.cleanupText(Contents.Text), Contents.Source, o.LimitGeneralEmbedding)
	if err != nil {
		return result, err
	}
	curContents := result.Contents[Contents.Id]
	// Cleanup previous keys
	for _, key := range curContents.Keys {
		llm.deleteRedisWildCard(llm.RedisClient.redisClient, key)
	}
	for _, key := range curContents.GeneralKeys {
		llm.deleteRedisWildCard(llm.RedisClient.redisClient, key)
	}

	// updating with new keys
	tmpGeneralKeys := append(curContents.GeneralKeys, generalKeys...)
	tmpKeys := append(curContents.GeneralKeys, tempKeys...)

	curContents = Contents
	curContents.GeneralKeys = tmpGeneralKeys
	curContents.Keys = tmpKeys

	result.Contents[Contents.Id] = curContents

	// Save the embedding data to Redis
	redisErr := llm.saveEmbeddingDataToRedis(result)
	return result, redisErr
}

// Load retrieves an embedding object from Redis storage based on a key.
//
// Parameters:
//   - client: Redis client instance for database operations.
//   - KeyID: The key used to retrieve the embedding object from Redis.
//
// Returns:
//   - error: An error if the key is not found or data cannot be unmarshalled.
func (llmEO *LLMEmbeddingObject) load(client *redis.Client, KeyID string) error {

	ctx := context.Background()

	// Retrieve data from Redis using the provided key
	data, err := client.Get(ctx, KeyID).Result()
	if err == redis.Nil {
		return fmt.Errorf("key not found")
	} else if err != nil {
		return err
	}

	// Unmarshal JSON data into the LLMEmbeddingObject structure
	err = json.Unmarshal([]byte(data), llmEO)
	if err != nil {
		return err
	}

	return nil
}

// func (llm *LLMContainer) LoadEmbeddData(Index string, options ...LLMCallOption) (map[string]LLMEmbeddingContent, error) {

// }
// Delete removes an embedding object from Redis storage based on a key.
//
// Parameters:
//   - rdb: Redis client instance for database operations.
//   - KeyID: The key used to delete the embedding object from Redis.
//
// Returns:
//   - error: An error if the key cannot be deleted or Redis connection fails.
func (llmEO LLMEmbeddingObject) delete(rdb *redis.Client, KeyID string) error {
	ctx := context.Background()
	// Check Redis connection
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		return err
	}
	// Delete the specified key from Redis
	_, err = rdb.Del(ctx, KeyID).Result()
	if err != nil {
		return err
	}
	return nil
}

// Save stores an embedding object in Redis after deleting any existing entry.
//
// Parameters:
//   - rdb: Redis client instance for database operations.
//   - KeyID: The key used to store the embedding object in Redis.
//
// Returns:
//   - error: An error if the save operation fails.
func (llmEO *LLMEmbeddingObject) save(rdb *redis.Client, KeyID string) error {
	ctx := context.TODO()

	// Delete any existing record before saving a new one
	llmEO.delete(rdb, llmEO.getRawDocRedisId())
	// Serialize the embedding object to JSON format
	data, err := json.Marshal(llmEO)
	if err != nil {
		return err
	}
	// Check Redis connection before proceeding
	_, err = rdb.Ping(ctx).Result()
	if err != nil {
		return err
	}
	// Store the serialized data in Redis with no expiration
	err = rdb.Del(ctx, KeyID).Err()
	if err != nil {
		return err
	}
	err = rdb.Set(ctx, KeyID, data, 0).Err()
	if err != nil {
		return err
	}
	return nil
}

// List retrieves multiple embedding objects from Redis with pagination support.
//
// Parameters:
//   - rdb: Redis client instance for database operations.
//   - KeyID: The prefix used to filter stored keys in Redis.
//   - offset: The starting position for retrieval.
//   - limit: The number of items to retrieve.
//
// Returns:
//   - map[string]interface{}: A map containing retrieved objects and total count.
//   - error: An error if the operation fails.
func (llm *LLMContainer)ListEmbeddings(KeyID string, offset, limit int) (map[string]interface{}, error){
	oe:= LLMEmbeddingObject{}
	return oe.list(llm.RedisClient.redisClient, KeyID, offset, limit)
}
func (llmEO LLMEmbeddingObject) list(rdb *redis.Client, KeyID string, offset, limit int) (map[string]interface{}, error) {
	ctx := context.Background()

	// Check Redis connection
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		return nil, err
	}
	// Retrieve all matching keys with the given prefix
	keys, err := rdb.Keys(ctx, KeyID+"*").Result()
	if err != nil {
		return nil, err
	}

	total := len(keys)
	if offset > total {
		offset = total
	}

	end := offset + limit
	if end > total {
		end = total
	}

	// Load embedding objects within the requested range
	var results []LLMEmbeddingObject
	for _, key := range keys[offset:end] {
		var obj LLMEmbeddingObject
		err := obj.load(rdb, key)
		if err == nil {
			results = append(results, obj)
		}
	}

	response := map[string]interface{}{
		"Rows":  results,
		"Total": total,
	}

	return response, nil
}

// saveEmbeddingDataToRedis saves the given embedding object to Redis.
//
// Parameters:
//   - obj: The embedding object to be stored in Redis.
//
// Returns:
//   - error: An error if the save operation fails.
func (llm *LLMContainer) saveEmbeddingDataToRedis(obj LLMEmbeddingObject) error {
	// Store the embedding object in Redis using its generated key
	return obj.save(llm.RedisClient.redisClient, obj.getRawDocRedisId())
}

// RemoveEmbedding deletes an embedding object and its associated keys from Redis.
//
// Parameters:
//   - ObjectId: The unique identifier for the embedding object to be removed.
//   - Index: The Index of the embedding object.
//
// Returns:
//   - error: An error if deletion fails.
func (llm *LLMContainer) RemoveEmbedding(Index string, options ...LLMCallOption) error {

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}
	llmo := LLMEmbeddingObject{
		EmbeddingPrefix: o.getEmbeddingPrefix(),
		Index:           Index,
	}
	// Load the embedding object from Redis
	llmo.load(llm.RedisClient.redisClient, llmo.getRawDocRedisId())

	// Delete all associated keys stored in Redis
	for _, content := range llmo.Contents {
		for _, key := range content.Keys {
			_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key)
			if err != nil {
				return err
			}
		}
		for _, key := range content.GeneralKeys {
			_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key)
			if err != nil {
				return err
			}
		}
	}

	// Remove the embedding object from Redis
	return llmo.delete(llm.RedisClient.redisClient, llmo.getRawDocRedisId())
}

func (llm *LLMContainer) RemoveEmbeddingSubKey(Index, rawDocID string, options ...LLMCallOption) error {

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}
	llmo := LLMEmbeddingObject{
		EmbeddingPrefix: o.getEmbeddingPrefix(),
		Index:           Index,
	}
	// Load the embedding object from Redis
	llmo.load(llm.RedisClient.redisClient, llmo.getRawDocRedisId())
	keyToDelete := llmo.Contents[rawDocID]
	// Delete all associated keys stored in Redis

	for _, key := range keyToDelete.Keys {
		_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key)
		if err != nil {
			return err
		}
	}
	for _, key := range keyToDelete.GeneralKeys {
		_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key)
		if err != nil {
			return err
		}
	}
	delete(llmo.Contents, rawDocID)
	if len(llmo.Contents) == 0 {
		//deleting the key if it was empty
		return llmo.delete(llm.RedisClient.redisClient, llmo.getRawDocRedisId())
	} else {
		// saving the embedding object to Redis
		return llmo.save(llm.RedisClient.redisClient, llmo.getRawDocRedisId())

	}
}
