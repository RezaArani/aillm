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
	"fmt"

	"github.com/redis/go-redis/v9"
)

// saveKey saves a key in Redis.
//
// Parameters:
//   - ctx: The context for the Redis operation.
//   - rdb: The Redis client instance.
//   - KeyID: The ID of the key to save.
//   - data: The data to save.
//
// Returns:
//   - error: An error if the operation fails.
func saveKey(ctx context.Context, rdb *redis.Client, KeyID string, data []byte) error {
	err := rdb.Do(ctx, "JSON.SET", KeyID, "$", string(data)).Err()
	if err != nil {
		return fmt.Errorf("error setting JSON in Redis: %v", err)
	}
	return nil
}

// deleteKey deletes a key in Redis.
//
// Parameters:
//   - ctx: The context for the Redis operation.
//   - rdb: The Redis client instance.
//   - KeyID: The ID of the key to delete.
func deleteKey(ctx context.Context, rdb *redis.Client, KeyID, indexName string) error {
	_, err := rdb.Do(ctx, "JSON.DEL", KeyID, "$").Result()
	if err != nil {
		return fmt.Errorf("error deleting JSON in Redis: %v", err)
	}
	err = rdb.Do(ctx, "FT.DEL", "rawDocsIdx:"+indexName, KeyID).Err()
	if err != nil {
		return err
	}

	return nil
}

// createIndex creates an index in Redis.
//
// Parameters:
//   - ctx: The context for the Redis operation.
//   - rdb: The Redis client instance.
//   - prefix: The prefix of the index.
//
// Returns:
//   - error: An error if the operation fails.
func createIndex(ctx context.Context, rdb *redis.Client, prefix string) error {
	indexName := "rawDocsIdx"
	if prefix != "" {
		indexName += ":" + prefix
	}
	_, err := rdb.Do(ctx, "FT.INFO", indexName).Result()
	if err != nil {
		// If the index does not exist, create it
		indexPrefix := "rawDocs:"
		if prefix != "" {
			indexPrefix += prefix + ":"
		}
		err = rdb.Do(ctx, "FT.CREATE", indexName,
			"ON", "JSON", // فعال‌سازی JSON
			"PREFIX", "1", indexPrefix, // پیشوند کلیدهای جستجو
			"SCHEMA",
			"$.Contents.*.GeneralKeys[*]", "AS", "GeneralKeys", "TAG",
			"$.Contents.*.Keys[*]", "AS", "Keys", "TAG",
		).Err()

		if err != nil {
			return fmt.Errorf("error creating index: %v", err)
		}
	}
	return nil
}
