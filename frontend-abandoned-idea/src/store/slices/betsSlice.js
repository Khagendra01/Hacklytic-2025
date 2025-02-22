import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  bets: [],
  loading: false,
  error: null,
  availableBets: [
    {
      id: 1,
      event: 'Manchester United vs Liverpool',
      type: 'Football',
      odds: { home: 2.1, draw: 3.2, away: 3.5 },
      date: '2024-03-15',
      time: '20:00',
    },
    {
      id: 2,
      event: 'Lakers vs Warriors',
      type: 'Basketball',
      odds: { home: 1.8, away: 2.1 },
      date: '2024-03-14',
      time: '19:30',
    },
    {
      id: 3,
      event: 'Djokovic vs Nadal',
      type: 'Tennis',
      odds: { player1: 1.9, player2: 1.95 },
      date: '2024-03-16',
      time: '15:00',
    },
  ],
};

const betsSlice = createSlice({
  name: 'bets',
  initialState,
  reducers: {
    setBets: (state, action) => {
      state.bets = action.payload;
    },
    addBet: (state, action) => {
      state.bets.unshift(action.payload);
    },
    updateBet: (state, action) => {
      const index = state.bets.findIndex(bet => bet.id === action.payload.id);
      if (index !== -1) {
        state.bets[index] = action.payload;
      }
    },
    setAvailableBets: (state, action) => {
      state.availableBets = action.payload;
    },
  },
});

export const { setBets, addBet, updateBet, setAvailableBets } = betsSlice.actions;
export default betsSlice.reducer;