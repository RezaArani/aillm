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
//   - Source: The origin of the content, such as a file name, URL, or other identifier.
//   - Keys: A slice of strings representing the Redis keys associated with this content.
type LLMEmbeddingContent struct {
	Text   string `json:"Text" redis:"Text"`
	Title  string `json:"Title" redis:"Title"`
	Source string `json:"Source" redis:"Source"`
	Keys   []string
}

// LLMEmbeddingObject represents a collection of embedded text contents grouped under a specific object ID.
//
// This struct serves as a container for multiple pieces of embedded text content, organized by language or context.
// It provides a way to store and manage embeddings for a specific use case or document.
//
// Fields:
//   - EmbeddingPrefix: A unique prefix or identifier for the embedding object (e.g., "ObjectId").
//   - Title: A title for the embedding object, providing a meaningful description.
//   - Contents: A map of language-specific content, where the key is the language code (e.g., "en", "pt")
//               and the value is an LLMEmbeddingContent struct containing the associated content details.
type LLMEmbeddingObject struct {
	EmbeddingPrefix string                         `json:"EmbeddingPrefix" redis:"EmbeddingPrefix"`
	Title           string                         `json:"Title" redis:"Title"`
	Contents        map[string]LLMEmbeddingContent `json:"Contents" redis:"Contents"`
}

// getRawDocRedisId generates a unique Redis key for storing raw document data.
// It combines the object ID and a sanitized version of the title to create a consistent key format.
//
// Returns:
//   - A string representing the Redis key in the format "rawDocs:ObjectId:Title".
func (llmeo LLMEmbeddingObject) getRawDocRedisId() string {
	// Construct Redis key using object ID and sanitized title
	return "rawDocs:" + llmeo.EmbeddingPrefix + ":" + llmeo.sanitizeRedisKey(llmeo.Title)
}

// EmbeddFile processes and embeds the content of a given file into the LLM system.
//
// Parameters:
//   - ObjectId: A unique identifier for the embedding object.
//   - Title: The title of the document being embedded. Also it will be used for the raw file redis key
//   - fileName: The path to the file to be embedded.
//   - tc: Configuration for transcription, such as language settings.
//
// Returns:
//   - LLMEmbeddingObject: The embedded object containing the processed content.
//   - error: An error if any issues occur during processing.
func (llm LLMContainer) EmbeddFile(Title, fileName string, tc TranscribeConfig, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	var result LLMEmbeddingObject
	EmbeddingContents := make(map[string]LLMEmbeddingContent)
	// Transcribe the file to extract text content
	fileContents, _, transcribeErr := llm.Transcriber.transcribeFile(fileName, "", tc)
	if transcribeErr != nil {
		return result, transcribeErr
	}

	// Store transcribed content with language as key
	EmbeddingContents[tc.Language] = LLMEmbeddingContent{
		Text:   fileContents,
		Title:  Title,
		Source: fileName,
	}

	// Embed the transcribed text into the LLM system
	embeddedTextObjects, embedErr := llm.EmbeddText(Title, EmbeddingContents, options...)
	if embedErr != nil {
		return result, embedErr
	}
	return embeddedTextObjects, embedErr
}

// EmbeddURL processes and embeds content from a given URL into the LLM system.
//
// Parameters:
//   - ObjectId: A unique identifier for the embedding object.
//   - Title: The title associated with the content being embedded.
//   - url: The web URL from which content will be transcribed and embedded.
//   - tc: Configuration for transcription, including language and extraction options.
//
// Returns:
//   - LLMEmbeddingObject: The embedded object containing the processed content.
//   - error: An error if any issues occur during the transcription or embedding process.
func (llm LLMContainer) EmbeddURL(Title, url string, tc TranscribeConfig, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	var result LLMEmbeddingObject
	EmbeddingContents := make(map[string]LLMEmbeddingContent)
	// Transcribe the content from the provided URL
	fileContents, _, transcribeErr := llm.Transcriber.transcribeURL(url, tc)
	if transcribeErr != nil {
		return result, transcribeErr
	}

	// Store transcribed content with the specified language as key
	EmbeddingContents[tc.Language] = LLMEmbeddingContent{
		Text:   fileContents,
		Title:  Title,
		Source: url,
	}

	// Embed the transcribed text into the LLM system
	// o := LLMCallOptions{}
	// for _, opt := range options {opt(&o)}
	embeddedTextObjects, embedErr := llm.EmbeddText(Title, EmbeddingContents, options...)
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
//   - Title: Title associated with the content being embedded.
//   - Contents: A map containing language-specific content to be embedded.
//
// Returns:
//   - LLMEmbeddingObject: The resulting embedding object after processing and storage.
//   - error: An error if any issues occur during embedding or Redis operations.
func (llm *LLMContainer) EmbeddText(Title string, Contents map[string]LLMEmbeddingContent, options ...LLMCallOption) (LLMEmbeddingObject, error) {

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}

	result := LLMEmbeddingObject{
		EmbeddingPrefix: o.getEmbeddingPrefix(),
		Title:    Title,
	}

	// Load existing data from Redis if available
	result.Load(llm.RedisClient.redisClient, result.getRawDocRedisId())

	// Delete existing Redis keys associated with the content
	for _, EmbeddingContents := range result.Contents {
		for _, key := range EmbeddingContents.Keys {
			llm.deleteRedisWildCard(llm.RedisClient.redisClient, key)

		}
	}

	// Store new contents
	result.Contents = Contents

	// Process embedding for each language in contents
	for language, content := range Contents {
		tempKeys, _, err := llm.embedText(o.getEmbeddingPrefix()+":"+language+":", llm.Transcriber.cleanupText(content.Text), content.Title, content.Source)
		if err != nil {
			return result, err
		}
		result.Contents[language] = LLMEmbeddingContent{
			Text:   content.Text,
			Source: content.Source,
			Title:  content.Title,
			Keys:   tempKeys,
		}
	}

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
func (llmEO *LLMEmbeddingObject) Load(client *redis.Client, KeyID string) error {

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

// Delete removes an embedding object from Redis storage based on a key.
//
// Parameters:
//   - rdb: Redis client instance for database operations.
//   - KeyID: The key used to delete the embedding object from Redis.
//
// Returns:
//   - error: An error if the key cannot be deleted or Redis connection fails.
func (llmEO LLMEmbeddingObject) Delete(rdb *redis.Client, KeyID string) error {
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
func (llmEO *LLMEmbeddingObject) Save(rdb *redis.Client, KeyID string) error {
	ctx := context.Background()

	// Delete any existing record before saving a new one
	llmEO.Delete(rdb, llmEO.getRawDocRedisId())
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
func (llmEO LLMEmbeddingObject) List(rdb *redis.Client, KeyID string, offset, limit int) (map[string]interface{}, error) {
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
		err := obj.Load(rdb, key)
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
	return obj.Save(llm.RedisClient.redisClient, obj.getRawDocRedisId())
}

// RemoveEmbeddingDataFromRedis deletes an embedding object and its associated keys from Redis.
//
// Parameters:
//   - ObjectId: The unique identifier for the embedding object to be removed.
//   - Title: The title of the embedding object.
//
// Returns:
//   - error: An error if deletion fails.
func (llm *LLMContainer) RemoveEmbeddingDataFromRedis(Title string, options ...LLMCallOption) error {

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}
	llmo := LLMEmbeddingObject{
		EmbeddingPrefix: o.getEmbeddingPrefix(),
		Title:    Title,
	}
	// Load the embedding object from Redis
	llmo.Load(llm.RedisClient.redisClient, llmo.getRawDocRedisId())

	// Delete all associated keys stored in Redis
	for _, content := range llmo.Contents {
		for _, key := range content.Keys {
			_, err := llm.deleteRedisWildCard(llm.RedisClient.redisClient, key)
			if err != nil {
				return err
			}
		}
	}

	// Remove the embedding object from Redis
	return llmo.Delete(llm.RedisClient.redisClient, llmo.getRawDocRedisId())
}
