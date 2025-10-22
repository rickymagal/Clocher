/**
 * @file test_tokenizer.c
 * @brief Unit tests for the tokenizer (size query, encode, decode invariants).
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>   /* malloc, free */
#include <stdint.h>   /* uint32_t     */
#include "ie_io.h"

/**
 * @brief Program entrypoint for tokenizer tests.
 *
 * Steps:
 * 1) Load vocab (file present or stub).
 * 2) Encode with size-only query to get the required buffer length.
 * 3) Encode into a real buffer and verify counts.
 * 4) Decode and check formatting invariants (starts with 'T', space count).
 *
 * @return 0 on success.
 */
int main(void) {
  ie_vocab_t v;
  assert(ie_vocab_load("models/gpt-oss-20b/vocab.json", &v) == 0);
  assert(v.vocab_size > 0);

  const char *txt = "hello world  from   engine";

  /* Size-only query */
  uint32_t needed = 0;
  assert(ie_tok_encode(&v, txt, NULL, &needed) == 0);
  assert(needed == 4);

  /* Full encode */
  uint32_t *ids = (uint32_t*)malloc(sizeof(uint32_t) * needed);
  assert(ids != NULL);
  uint32_t got = needed; /* capacity in, count out */
  assert(ie_tok_encode(&v, txt, ids, &got) == 0);
  assert(got == needed);

  /* Decode to text */
  char buf[256];
  assert(ie_tok_decode(&v, ids, got, buf, sizeof(buf)) == 0);

  /* Sanity: decoding starts with a "T" token placeholder and contains spaces between tokens */
  assert(strncmp(buf, "T", 1) == 0);

  /* There should be exactly (needed - 1) spaces */
  unsigned spaces = 0;
  for (const char *p = buf; *p; ++p) if (*p == ' ') ++spaces;
  assert(spaces == (needed > 0 ? needed - 1 : 0));

  free(ids);

  printf("ok test_tokenizer\n");
  return 0;
}
