const { VertexAI } = require('@google-cloud/vertexai');
const functions = require('@google-cloud/functions-framework');

// Initialize Vertex AI outside the handler for better performance
const vertex_ai = new VertexAI({ 
  project: 'new-man-app', 
  location: 'us-central1' 
});

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

    // Use gemini-2.5-flash-lite as primary, with fallbacks
    const modelNames = [
      'gemini-2.5-flash-lite',  // Primary model
      'gemini-2.0-flash-lite',  // Fallback 1
      'gemini-1.5-flash-lite',  // Fallback 2
      'gemini-1.5-flash',       // Fallback 3
      'gemini-pro'              // Final fallback
    ];

    let analysisText = '';
    let lastError = null;
    let successfulModel = '';

    // Try each model until one works
    for (const modelName of modelNames) {
      try {
        console.log(`Trying model: ${modelName}`);
        
        const generativeModel = vertex_ai.getGenerativeModel({
          model: modelName,
          generationConfig: {
            maxOutputTokens: 1024,
            temperature: 0.7,
            topP: 0.8,
            topK: 40
          },
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
        });

        const scoreForPrompt = (typeof virtueScore === 'number' && !isNaN(virtueScore))
          ? virtueScore.toFixed(1)
          : 'N/A';

        // Optimized prompt for flash-lite models
        const prompt = `You are Astrid, a wise and empathetic virtual sponsor for people in recovery. Be supportive, non-judgmental, and encouraging.

Virtue being assessed: ${virtueName}
Definition: ${virtueDef}
User's self-assessment score: ${scoreForPrompt}/10
Related character defects noted:
${defectDetails || 'No specific defect details provided'}

Provide a compassionate analysis (about 200 words). Speak directly to the user ("You..."). Acknowledge their courage in self-reflection. Gently connect their specific defect ratings to this virtue's development. Offer hopeful encouragement for growth in this area. Focus on strength and possibility.`;

        const result = await generativeModel.generateContent(prompt);
        const response = result.response;
        
        if (response.candidates && response.candidates[0] && response.candidates[0].content) {
          analysisText = response.candidates[0].content.parts[0].text;
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