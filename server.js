import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { HfInference } from '@huggingface/inference';
import rateLimit from 'express-rate-limit';

dotenv.config();
const app = express();

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100
});

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(limiter);

// Initialize Hugging Face client
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({ 
        status: 'ok', 
        timestamp: new Date().toISOString() 
    });
});

// Chat endpoint
app.post('/api/chat', async (req, res) => {
    try {
        const { message, image_url } = req.body;
        
        // Message validation
        if (!message || typeof message !== 'string') {
            return res.status(400).json({
                error: 'Message is required and must be a string',
                timestamp: new Date().toISOString()
            });
        }

        // Log request for debugging
        console.log('Received request:', {
            message,
            image_url,
            timestamp: new Date().toISOString()
        });

        const generated = await hf.textGeneration({
            model: "deepseek-ai/deepseek-coder-33b-instruct",
            inputs: message,
            parameters: {
                max_new_tokens: 1000,
                temperature: 0.7,
                return_full_text: false,
                do_sample: true,
                top_p: 0.95,
                top_k: 50,
                repetition_penalty: 1.1,
                length_penalty: 1.0,
                stop: ["</s>", "Human:", "Assistant:"]
            }
        });

        if (!generated || !generated.generated_text) {
            throw new Error('No response generated from the model');
        }

        res.json({
            response: {
                message: generated.generated_text,
                model: "deepseek-ai/deepseek-coder-33b-instruct",
                timestamp: new Date().toISOString()
            }
        });
    } catch (error) {
        console.error('Error details:', error);
        res.status(500).json({
            error: 'Error processing request',
            details: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Define porta para o Render
const PORT = process.env.PORT || 3000;

// Log para debug
console.log('Starting server with PORT:', PORT);

// Inicializa o servidor
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on port ${PORT} - ${new Date().toISOString()}`);
});
