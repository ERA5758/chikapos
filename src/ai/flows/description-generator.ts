/**
 * @fileOverview An AI agent for generating product descriptions.
 */

import { ai } from '@/ai/genkit';
import { z } from 'genkit';

export const DescriptionGeneratorInputSchema = z.object({
  productName: z.string().describe('The name of the product.'),
  category: z.string().describe('The category of the product (e.g., "Pakaian", "Elektronik", "Buku").'),
  topSellingProducts: z.array(z.string()).describe('A list of other top-selling products for context.'),
});
export type DescriptionGeneratorInput = z.infer<typeof DescriptionGeneratorInputSchema>;

export const DescriptionGeneratorOutputSchema = z.object({
  description: z.string().describe('A concise, attractive product description in Indonesian (2-3 sentences).'),
});
export type DescriptionGeneratorOutput = z.infer<typeof DescriptionGeneratorOutputSchema>;

const promptText = `Anda adalah seorang copywriter yang ahli untuk merek "Chika".
Tugas Anda adalah membuat deskripsi produk yang singkat (2-3 kalimat), menarik, dan persuasif untuk item berikut.

Gunakan Bahasa Indonesia.

Detail Produk:
- Nama Produk: {{productName}}
- Kategori: {{category}}

Sebagai konteks, produk terlaris lainnya di toko ini adalah: {{#each topSellingProducts}}{{{this}}}{{#unless @last}}, {{/unless}}{{/each}}.

Fokus pada fitur, manfaat, dan keunikan produk. Buat pelanggan ingin segera memilikinya.

Contoh untuk "Kaos Polos Katun":
"Dibuat dari 100% katun premium, kaos ini menawarkan kenyamanan sepanjang hari dan gaya yang tak lekang oleh waktu. Sempurna untuk tampilan kasual maupun sebagai lapisan dasar, jadikan ini sebagai andalan baru di lemari Anda."

Hasilkan deskripsi untuk {{productName}} dan kembalikan dalam format JSON yang valid.`;

export const descriptionGeneratorFlow = ai.defineFlow(
  {
    name: 'descriptionGeneratorFlow',
    inputSchema: DescriptionGeneratorInputSchema,
    outputSchema: DescriptionGeneratorOutputSchema,
  },
  async (input) => {
    const { output } = await ai.generate({
      model: 'openai/gpt-4o',
      prompt: promptText,
      input: input,
      output: {
        schema: DescriptionGeneratorOutputSchema,
      },
    });
    
    if (!output) {
      throw new Error('AI did not return a valid description.');
    }
    return output;
  }
);

export async function generateDescription(
  input: DescriptionGeneratorInput
): Promise<DescriptionGeneratorOutput> {
    return descriptionGeneratorFlow(input);
}
