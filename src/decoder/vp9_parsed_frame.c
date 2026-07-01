/**
 * compute-vp9 — parsed frame structure lifecycle helpers
 */
#include "vp9_parsed_frame.h"
#include <stdlib.h>
#include <string.h>

vp9_parsed_frame_t *vp9_parsed_frame_alloc(uint32_t width, uint32_t height)
{
    vp9_parsed_frame_t *pf = calloc(1, sizeof(*pf));
    if (!pf) return NULL;

    pf->mv_grid_width  = (width + 3) / 4;
    pf->mv_grid_height = (height + 3) / 4;
    
    pf->mv_grid = calloc(pf->mv_grid_width * pf->mv_grid_height, sizeof(cvp9_mv_t));
    if (!pf->mv_grid) {
        free(pf);
        return NULL;
    }

    /* Initial capacities */
    pf->blocks_capacity = 256;
    pf->blocks = malloc(pf->blocks_capacity * sizeof(*pf->blocks));
    
    pf->coeffs_capacity = 4096;
    pf->coeffs = malloc(pf->coeffs_capacity * sizeof(*pf->coeffs));

    if (!pf->blocks || !pf->coeffs) {
        vp9_parsed_frame_free(pf);
        return NULL;
    }

    return pf;
}

void vp9_parsed_frame_free(vp9_parsed_frame_t *pf)
{
    if (!pf) return;
    free(pf->blocks);
    free(pf->coeffs);
    free(pf->mv_grid);
    free(pf);
}

void vp9_parsed_frame_reset(vp9_parsed_frame_t *pf)
{
    if (!pf) return;
    pf->num_blocks = 0;
    pf->num_coeffs = 0;
    memset(pf->mv_grid, 0, pf->mv_grid_width * pf->mv_grid_height * sizeof(cvp9_mv_t));
    memset(&pf->hdr, 0, sizeof(pf->hdr));
}

bool vp9_parsed_frame_ensure_blocks(vp9_parsed_frame_t *pf, uint32_t count)
{
    if (pf->num_blocks + count <= pf->blocks_capacity) return true;
    
    uint32_t new_cap = pf->blocks_capacity * 2;
    while (pf->num_blocks + count > new_cap) new_cap *= 2;
    
    vp9_macroblock_info_t *new_blocks = realloc(pf->blocks, new_cap * sizeof(*new_blocks));
    if (!new_blocks) return false;
    
    pf->blocks = new_blocks;
    pf->blocks_capacity = new_cap;
    return true;
}

bool vp9_parsed_frame_ensure_coeffs(vp9_parsed_frame_t *pf, uint32_t count)
{
    if (pf->num_coeffs + count <= pf->coeffs_capacity) return true;
    
    uint32_t new_cap = pf->coeffs_capacity * 2;
    while (pf->num_coeffs + count > new_cap) new_cap *= 2;
    
    int16_t *new_coeffs = realloc(pf->coeffs, new_cap * sizeof(*new_coeffs));
    if (!new_coeffs) return false;
    
    pf->coeffs = new_coeffs;
    pf->coeffs_capacity = new_cap;
    return true;
}
