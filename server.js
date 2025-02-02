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

        // Prepare message content
        const content = [
            {
                type: "text",
                text: message
            }
        ];

        // Add image if provided and valid
        if (image_url && typeof image_url === 'string') {
            content.push({
                type: "image_url",
                image_url: { url: image_url }
            });
        }

        // Call to Hugging Face API using latest DeepSeek model
        const generated = await hf.textGeneration({
            model: "deepseek-ai/deepseek-coder-33b-instruct",  // Updated to latest DeepSeek model
            inputs: message,
            parameters: {
                max_new_tokens: 1000,        // Increased token limit
                temperature: 0.7,
                return_full_text: false,
                do_sample: true,
                top_p: 0.95,
                top_k: 50,
                repetition_penalty: 1.1,    // Added to improve response quality
                length_penalty: 1.0,        // Added to balance response length
                stop: ["</s>", "Human:", "Assistant:"]  // Added stop tokens
            }
        });

        // Log response for debugging
        console.log('HF Response:', generated);

        // Verify response contains generated text
        if (!generated || !generated.generated_text) {
            throw new Error('No response generated from the model');
        }

        // Return structured response
        res.json({
            response: {
                message: generated.generated_text,
                model: "deepseek-ai/deepseek-coder-33b-instruct",
                timestamp: new Date().toISOString()
            }
        });
    } catch (error) {
        // Detailed error logging
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            stack: error.stack,
            timestamp: new Date().toISOString()
        });

        // Determine appropriate status code
        const statusCode = error.name === 'ValidationError' ? 400 : 500;

        // Return structured error
        res.status(statusCode).json({
            error: 'Error processing request',
            details: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Error handling for uncaught errors
app.use((err, req, res, next) => {
    console.error('Global error handler:', {
        error: err.stack,
        timestamp: new Date().toISOString()
    });
    
    res.status(500).json({
        error: 'Something went wrong!',
        details: err.message,
        timestamp: new Date().toISOString()
    });
});

// Server initialization
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT} - ${new Date().toISOString()}`);
});
