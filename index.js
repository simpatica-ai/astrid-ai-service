const { GoogleGenAI } = require('@google/genai');
const functions = require('@google-cloud/functions-framework');

// ---- Vertex endpoint configuration (edit here for a regional/BAA switch) ----
const GCP_PROJECT = 'new-man-app';
const GCP_LOCATION = 'global'; // global endpoint required for Gemini 3.x; switch to a region (e.g. 'us-central1') for data residency / BAA
// Model fallback chain (global endpoint). Primary first.
const MODEL_NAMES = [
  'gemini-3.5-flash',      // Primary
  'gemini-3.1-flash-lite', // Fallback 1
  'gemini-2.5-flash',      // Fallback 2 — temporary bridge until 2.5 retires
];
// ---------------------------------------------------------------------------

// Initialize the GenAI client outside the handler for better performance
const ai = new GoogleGenAI({ vertexai: true, project: GCP_PROJECT, location: GCP_LOCATION });

functions.http('getAstridAnalysis', async (req, res) => {
  // Set CORS headers
  res.set('Access-Control-Allow-Origin', '*');
  res.set('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.set('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.status(204).send('');
    return;
  }

  try {
    // Validate request body
    if (!req.body || typeof req.body !== 'object') {
      return res.status(400).send({ error: 'Invalid request body' });
    }

    const { virtueName, virtueDef, virtueScore, defectDetails } = req.body;
    
    // Validate required fields
    if (!virtueName || !virtueDef) {
      return res.status(400).send({ error: 'Missing required fields: virtueName and virtueDef are required' });
    }

    let analysisText = '';
    let lastError = null;
    let successfulModel = '';

    // Try each model until one works
    for (const modelName of MODEL_NAMES) {
      try {
        console.log(`Trying model: ${modelName}`);

        const scoreForPrompt = (typeof virtueScore === 'number' && !isNaN(virtueScore))
          ? virtueScore.toFixed(1)
          : 'N/A';

        // Optimized prompt for flash-lite models
        const prompt = `
                  Analyze the following data for a user reviewing assessment results intended to identify character defects that preempt being virtuous. Your tone must be direct, objective, and insightful. The goal is a qualitative analysis of the user's defects to generate a customized prompt for development. This is the first stage where the user undertakes dismantling of these character defects. Refer to the user as "you" and be empathetic but firm in your analysis.

                  **SPELLING AND GRAMMAR ARE VERY IMPORTANT.  Review all responses for spelling and grammar.

          DATA:
          - Virtue: ${virtueName}
          - User's Score: ${scoreForPrompt}/10 (A lower score indicates a greater challenge) A score in the lower third indicates deep challenges with this virtue, a score in the middle third indicates moderate challenges as well as capacity with this virtue, and within the upper third indicates strong capacity with this virtue and likely only minor challenges.)
          - Associated Defect Ratings:
          ${defectDetails || 'No specific defect details provided'}

          TASKS:
          1.  Briefly synthesize the user's main challenge regarding this virtue based on the frequency and harm of the associated defects.
          2.  Identify the most significant defect(s) contributing to the low score.
          3.  Conclude with a direct, actionable prompt for the user to reflect on for developing this virtue. Frame it as a question or a statement for consideration.
          4.  Keep the entire response under 150 words.`;

        const result = await ai.models.generateContent({
          model: modelName,
          contents: prompt,
          config: {
            maxOutputTokens: 1024,
            temperature: 0.6,
            topP: 0.8,
            topK: 40,
            safetySettings: [
              {
                category: 'HARM_CATEGORY_HATE_SPEECH',
                threshold: 'BLOCK_MEDIUM_AND_ABOVE'
              },
              {
                category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold: 'BLOCK_MEDIUM_AND_ABOVE'
              },
              {
                category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold: 'BLOCK_MEDIUM_AND_ABOVE'
              },
              {
                category: 'HARM_CATEGORY_HARASSMENT',
                threshold: 'BLOCK_MEDIUM_AND_ABOVE'
              }
            ]
          }
        });

        const text = result.text ?? result.candidates?.[0]?.content?.parts?.[0]?.text;
        if (text && text.trim().length > 0) {
          analysisText = text;
          successfulModel = modelName;
          console.log(`Success with model: ${modelName}`);
          break;
        } else {
          throw new Error('Invalid response format from model');
        }
        
      } catch (error) {
        lastError = error;
        console.warn(`Model ${modelName} failed:`, error.message);
        continue; // Try next model
      }
    }

    if (!analysisText) {
      console.error('All models failed:', lastError);
      // Provide a meaningful fallback response
      analysisText = `I see you're reflecting on ${virtueName}, which involves ${virtueDef.toLowerCase()}. Your willingness to engage in this self-assessment shows real courage and commitment to growth. 

Based on your reflections, this appears to be an area where focused attention could bring meaningful progress. Remember that every step forward counts, and the awareness you're building today is the foundation for tomorrow's growth. You have the strength to develop this quality in your life.`;
    }

    res.status(200).send({ 
      analysis: analysisText,
      model: successfulModel || 'fallback',
      success: true 
    });

  } catch (error) {
    console.error('Unexpected error in getAstridAnalysis:', error);
    res.status(500).send({ 
      error: 'Internal server error',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});